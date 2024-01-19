
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from datetime import datetime
from functools import partial
from typing import Any, Dict, List

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq)

from rl4lms.data_pools.text_generation_pool import Sample
from rl4lms.envs.text_generation.env import TextGenEnv
from rl4lms.envs.text_generation.evaluation_utils import evaluate_on_samples
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.registry import (DataPoolRegistry,
                                                  MetricRegistry,
                                                  RewardFunctionRegistry,
                                                  PolicyRegistry,
                                                  AlgorithmRegistry,
                                                  WrapperRegistry)
from rl4lms.envs.text_generation.reward import RewardFunction
from rl4lms.envs.text_generation.utils_supervised import evaluate_on_samples as evaluate_supervised
from rl4lms.envs.text_generation.utils_supervised import (get_datasets_for_causal,
                                                          get_datasets_for_seq2seq,
                                                          tokenize_causal,
                                                          tokenize_seq2seq,
                                                          EvalCallack)
from rl4lms.envs.text_generation.warm_start import TrainerWarmStartMixin


def build_tokenizer(tokenizer_config: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["model_name"])
    if tokenizer.pad_token is None and tokenizer_config.get("pad_token_as_eos_token", True):
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = tokenizer_config.get("padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get("truncation_side", "left")
    return tokenizer


def build_reward_fn(reward_config: Dict[str, Any]):
    reward_fn = RewardFunctionRegistry.get(reward_config["id"],
                                           reward_config.get("args", {}))
    return reward_fn


def build_metrics(metric_configs: List[Dict[str, Any]]):
    metrics = [MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
               for metric_config in metric_configs]
    return metrics


def build_datapool(datapool_config: Dict[str, Any]):
    def _get_datapool_by_split(split: str):
        kwargs = datapool_config.get("args", {})
        kwargs["split"] = split
        dp_split = DataPoolRegistry.get(datapool_config["id"], kwargs)
        return dp_split

    train_datapool = _get_datapool_by_split("train")
    frac_train_samples = datapool_config.get("frac_train_samples", -1)
    rnd_seed = datapool_config.get("train_samples_rnd_seed", -1)
    train_samples = __subsample_train_datapool(train_datapool, frac_train_samples, rnd_seed)

    val_datapool = _get_datapool_by_split("val")
    test_datapool = _get_datapool_by_split("test")

    samples_by_split = {
        "train": train_samples,
        "val": [sample for sample, _ in val_datapool],
        "test": [sample for sample, _ in test_datapool]
    }
    return samples_by_split


def __subsample_train_datapool(datapool, frac_train_samples: float, rnd_seed: int):
    if frac_train_samples <= 0 or frac_train_samples > 1:
        return [(sample, weight) for sample, weight in datapool]

    num_train_samples_to_choose = int(len(datapool) * frac_train_samples)
    indices_perm = np.random.RandomState(seed=rnd_seed).permutation(len(datapool)) if rnd_seed >= 0 else np.random.permutation(len(datapool))
    chosen_indices = indices_perm[:num_train_samples_to_choose]
    return [datapool[i] for i in chosen_indices]


def build_env(env_config: Dict[str, Any],
              reward_fn: RewardFunction,
              tokenizer: AutoTokenizer,
              train_samples: List[Sample]):
    # vectoried env
    env_kwargs = {
        "reward_function": reward_fn,
        "tokenizer": tokenizer,
        "samples": train_samples,
    }
    env_kwargs = {**env_kwargs, **env_config.get("args", {})}
    env = make_vec_env(TextGenEnv,
                       n_envs=env_config.get("n_envs", 1),
                       vec_env_cls=SubprocVecEnv,
                       env_kwargs=env_kwargs)
    return env


def build_alg(alg_config: Dict[str, Any],
              env: TextGenEnv,
              tracker: Tracker,
              policy_state: Dict[str, Any],
              alg_state: Dict[str, Any]):
    # TBD - move these to a registry once the experimentation is done
    # Also switch to Sb3 algos when possible with minimal code adaptations
    policy_config = alg_config["policy"]
    policy_cls = PolicyRegistry.get(policy_config["id"])
    alg_cls = AlgorithmRegistry.get(alg_config["id"])

    policy_args = policy_config["args"]
    policy_args["state_dict"] = policy_state
    alg_kwargs = {
        "policy": policy_cls,
        "env": env,
        "policy_kwargs": policy_args,
    }
    alg_kwargs = {**alg_kwargs, **alg_config.get("args")}
    wrapper = WrapperRegistry.get(alg_config["id"])
    alg = wrapper(alg_cls, alg_kwargs,
                  alg_config["kl_div"]["coeff"], tracker,
                  alg_config["kl_div"].get("target_kl", None),
                  alg_config["kl_div"].get("norm_reward", False))
    alg.load_from_dict(alg_state)
    return alg


class OnPolicyTrainer(TrainerWarmStartMixin):
    """
    A generic trainer for training LMs with onpolicy algorithms from SB3
    """

    def __init__(self,
                 tokenizer_config: Dict[str, Any],
                 datapool_config: Dict[str, Any],
                 reward_config: Dict[str, Any],
                 env_config: Dict[str, Any],
                 on_policy_alg_config: Dict[str, Any],
                 train_eval_config: Dict[str, Any],
                 tracker: Tracker = None,
                 experiment_name: str = '',
                 no_eval_on_train_set: bool = False
                 ):
        self._tokenizer_config = tokenizer_config
        self._datapool_config = datapool_config
        self._reward_config = reward_config
        self._env_config = env_config
        self._on_policy_alg_config = on_policy_alg_config
        self._train_eval_config = train_eval_config
        self._tracker = tracker
        self._experiment_name = experiment_name
        self._no_eval_on_train_set = no_eval_on_train_set
        self._start_and_end_eval_splits = ["val", "test"] if self._no_eval_on_train_set else ["train", "val", "test"]

        # Prevent HuggingFace error when temperature is int instead of float
        eval_gen_kwargs = self._train_eval_config.get("generation_kwargs", {})
        if eval_gen_kwargs is not None and "temperature" in eval_gen_kwargs:
            eval_gen_kwargs["temperature"] = float(eval_gen_kwargs["temperature"])

        on_policy_gen_kwargs = self._on_policy_alg_config["policy"]["args"]["generation_kwargs"]
        if on_policy_gen_kwargs is not None and "temperature" in on_policy_gen_kwargs:
            on_policy_gen_kwargs["temperature"] = float(on_policy_gen_kwargs["temperature"])

        self._setup()

    def _setup(self):
        # load trainer state from available previous checkpoint if available
        self.load_trainer_state(self._tracker)

        # build components
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._reward_fn = build_reward_fn(self._reward_config)
        self._metrics = build_metrics(self._train_eval_config.get("metrics", []))
        self._samples_by_split = build_datapool(self._datapool_config)
        self._env = build_env(self._env_config, self._reward_fn,
                              self._tokenizer, self._samples_by_split["train"])
        self._alg = build_alg(self._on_policy_alg_config,
                              self._env, self._tracker,
                              self._policy_state_dict,
                              self._alg_state_dict)

        # extract train params
        self._max_episode_length = self._env_config["args"]["max_episode_length"]
        self._max_prompt_length = self._env_config["args"]["max_prompt_length"]
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]
        self._n_iters = int(self._train_eval_config["n_iters"])
        self._n_steps_per_iter = self._env.num_envs * self._alg.n_steps

        # gen kwargs for evaluation (if it is different from rollout gen kwargs)
        self._eval_gen_kwargs = self._train_eval_config.get("generation_kwargs", None)

    def _evaluate_on_datapools(self, epoch: int,
                               splits: List[str] = ["val", "test"]):
        for split in splits:
            gen_kwargs = self._eval_gen_kwargs if split != "train" else self._on_policy_alg_config["policy"]["args"]["generation_kwargs"]

            # Prevent HuggingFace error when temperature is int instead of float
            if gen_kwargs is not None and "temperature" in gen_kwargs:
                gen_kwargs["temperature"] = float(gen_kwargs["temperature"])

            samples = self._samples_by_split[split]
            if isinstance(samples[0], tuple):
                samples = [sample for sample, _ in samples]

            evaluate_on_samples(policy=self._alg.policy,
                                tokenizer=self._tokenizer,
                                samples=samples,
                                batch_size=self._eval_batch_size,
                                max_prompt_length=self._max_prompt_length,
                                metrics=self._metrics,
                                epoch=epoch,
                                split_name=split,
                                tracker=self._tracker,
                                gen_kwargs=gen_kwargs)

    def train_and_eval(self):
        # evaluate on val and test set before fine-tuning once
        iter_start = self._trainer_state["current_iter"]
        self._evaluate_on_datapools(epoch=iter_start, splits=self._start_and_end_eval_splits)

        # train for given number of iters
        for epoch in range(iter_start, self._n_iters):
            start_time = datetime.utcnow()
            print(f"Starting train phase of epoch {epoch} out of {self._n_iters}")

            # current state
            self._trainer_state["current_iter"] = epoch

            # inner rollout and learn loop for on-policy algorithm
            self._alg.learn(self._n_steps_per_iter)

            time_delta = datetime.utcnow() - start_time
            print(f"Finished train phase of epoch {epoch} out of {self._n_iters}. Time took: {time_delta}")

            # save the policy checkpoint
            if (epoch + 1) % self._train_eval_config.get("save_every", 20) == 0:
                self.save_trainer_state(self._tracker, self._alg.policy, self._trainer_state)

            # evaluate on val set in the given intervals
            if (epoch + 1) % self._train_eval_config["eval_every"] == 0:
                start_time = datetime.utcnow()
                print(f"Starting validation phase of epoch {epoch} out of {self._n_iters}")

                self._evaluate_on_datapools(epoch=epoch, splits=["val"])

                time_delta = datetime.utcnow() - start_time
                print(f"Finished validation phase of epoch {epoch} out of {self._n_iters}. Time took: {time_delta}")

        # finally evaluate on val and test samples
        self._evaluate_on_datapools(epoch=epoch, splits=self._start_and_end_eval_splits)

        # save model here - we save only the language model
        if self._tracker is not None:
            self._tracker.save_auto_model(self._alg.policy.get_language_model())


class SupervisedTrainer:
    """
    A supervised trainer to train LMs (causal and seq2seq) on text generation tasks (wrapper on HF trainer)
    """

    def __init__(self,
                 tokenizer_config: Dict[str, Any],
                 datapool_config: Dict[str, Any],
                 train_eval_config: Dict[str, Any],
                 alg_config: Dict[str, Any],
                 tracker: Tracker = None,
                 no_eval_on_train_set: bool = False
                 ):
        self._tokenizer_config = tokenizer_config
        self._datapool_config = datapool_config
        self._train_eval_config = train_eval_config
        self._alg_config = alg_config
        self._tracker = tracker
        self._no_eval_on_train_set = no_eval_on_train_set
        self._start_and_end_eval_splits = ["val", "test"] if self._no_eval_on_train_set else ["train", "val", "test"]

        # Prevent HuggingFace error when temperature is int instead of float
        if self._train_eval_config is not None and "temperature" in self._train_eval_config["generation_kwargs"]:
            self._train_eval_config["generation_kwargs"]["temperature"] = float(self._train_eval_config["generation_kwargs"]["temperature"])

        self._setup()

    def _evaluate_on_datapools(self, epoch: int,
                               splits: List[str] = ["val", "test"]):
        for split in splits:
            gen_kwargs = self._gen_kwargs if split != "train" else self._alg_config["generation_kwargs"]

            # Prevent HuggingFace error when temperature is int instead of float
            if gen_kwargs is not None and "temperature" in gen_kwargs:
                gen_kwargs["temperature"] = float(gen_kwargs["temperature"])

            samples = self._samples_by_split[split]
            if isinstance(samples[0], tuple):
                samples = [sample for sample, _ in samples]

            evaluate_supervised(model=self._model,
                                tokenizer=self._tokenizer,
                                samples=samples,
                                batch_size=self._eval_batch_size,
                                max_prompt_length=self._max_prompt_length,
                                metrics_config_dict=self._metrics_config_dict,
                                epoch=epoch,
                                split_name=split,
                                tracker=self._tracker,
                                generation_kwargs=gen_kwargs
                                )

    def _setup(self):
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._metrics_config_dict = self._train_eval_config.get("metrics")
        self._samples_by_split = build_datapool(self._datapool_config)
        self._train_dataset = get_datasets_for_causal(self._samples_by_split["train"]) if self._alg_config["model_type"] == "causal" else \
            get_datasets_for_seq2seq(self._samples_by_split["train"])

        preprocess_fn = tokenize_causal if self._alg_config["model_type"] == "causal" else tokenize_seq2seq
        preprocess_fn = partial(preprocess_fn, tokenizer=self._tokenizer)
        self._tokenized_dataset = self._train_dataset.map(
            preprocess_fn, batched=True,
            remove_columns=self._train_dataset.column_names)
        model_cls = AutoModelForCausalLM if self._alg_config["model_type"] == "causal" else AutoModelForSeq2SeqLM

        self._gen_kwargs = self._train_eval_config["generation_kwargs"]
        self._model = model_cls.from_pretrained(self._alg_config["model_name"])
        if torch.cuda.device_count() > 0:
            self._model.parallelize()
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]

        # setting max prompt length
        self._max_prompt_length = self._tokenizer_config.get("max_length", self._tokenizer.model_max_length)

        if (self._alg_config["model_type"] == "causal") and (
                (self._max_prompt_length + self._gen_kwargs["max_new_tokens"]) > self._tokenizer.model_max_length):
            self._max_prompt_length = self._max_prompt_length - \
                                      self._gen_kwargs["max_new_tokens"]

        self._eval_callback = EvalCallack(self._samples_by_split["val"],
                                          self._gen_kwargs,
                                          self._eval_batch_size,
                                          self._tokenizer,
                                          self._metrics_config_dict,
                                          self._max_prompt_length,
                                          self._tracker)
        train_args = self._alg_config["training_args"]
        train_args["output_dir"] = self._tracker.checkpoint_base_path
        train_args["seed"] = np.random.randint(1e+2)  # random seed
        self._train_args = TrainingArguments(**train_args, report_to=["none"] if not self._tracker.wandb_log else ["wandb"])
        data_collator = DataCollatorForLanguageModeling(self._tokenizer, mlm=False) if self._alg_config["model_type"] == "causal" \
            else DataCollatorForSeq2Seq(self._tokenizer, self._model)

        self._trainer = Trainer(model=self._model,
                                tokenizer=self._tokenizer,
                                args=self._train_args,
                                data_collator=data_collator,
                                train_dataset=self._tokenized_dataset,
                                callbacks=[self._eval_callback])

    def train_and_eval(self):
        # evaluate on train, val, and test set before fine-tuning once
        self._evaluate_on_datapools(epoch=0, splits=self._start_and_end_eval_splits)

        # train using HF trainer
        self._trainer.train()

        # finally evaluate on train, val, and test samples
        self._evaluate_on_datapools(epoch=self._train_args.num_train_epochs, splits=self._start_and_end_eval_splits)

        # save model here - we save only the language model
        if self._tracker is not None:
            self._tracker.save_auto_model(self._model)
