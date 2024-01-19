
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, PreTrainedModel, AutoTokenizer)

from rl4lms.data_pools.custom_text_generation_pools import Sample
from rl4lms.envs.text_generation.registry import DataPoolRegistry
from rl4lms.envs.text_generation.registry import MetricRegistry
from rl4lms.envs.text_generation.training_utils import build_tokenizer
from rl4lms.envs.text_generation.utils_supervised import generate_on_samples


def get_train_samples(datapool_config):
    kwargs = datapool_config.get("args", {})
    kwargs["split"] = "train"
    train_datapool = DataPoolRegistry.get(datapool_config["id"], kwargs)

    num_train_samples = datapool_config.get("num_train_samples", -1)
    rnd_seed = datapool_config.get("train_samples_rnd_seed", -1)

    if num_train_samples <= 0:
        return [sample for sample, weight in train_datapool], torch.arange(len(train_datapool))

    indices_perm = np.random.RandomState(seed=rnd_seed).permutation(len(train_datapool)) if rnd_seed >= 0 else np.random.permutation(
        len(train_datapool))
    indices_to_use = indices_perm[:num_train_samples]
    return [train_datapool[i][0] for i in indices_to_use], torch.tensor(indices_to_use)


def setup_tokenizer_and_model(tokenizer_config, alg_config):
    tokenizer = build_tokenizer(tokenizer_config)
    model_cls = AutoModelForCausalLM if alg_config["model_type"] == "causal" else AutoModelForSeq2SeqLM

    model = model_cls.from_pretrained(alg_config["model_name"])
    if torch.cuda.device_count() > 0:
        model.parallelize()

    return tokenizer, model


def get_max_prompt_length(tokenizer_config, tokenizer, alg_config, gen_kwargs):
    max_prompt_length = tokenizer_config.get("max_length", tokenizer.model_max_length)

    if (alg_config["model_type"] == "causal") and ((max_prompt_length + gen_kwargs["max_new_tokens"]) > tokenizer.model_max_length):
        max_prompt_length = max_prompt_length - gen_kwargs["max_new_tokens"]

    return max_prompt_length


def __compute_metrics(metrics_config_dict: List[Dict[str, Any]],
                      samples: List[Sample],
                      all_prompt_texts: List[str],
                      all_generated_texts: List[str],
                      all_ref_texts: List[str],
                      all_meta_infos: List[Dict[str, Any]],
                      model: PreTrainedModel):
    # compute metrics
    metric_values = defaultdict(list)
    if metrics_config_dict is not None:
        for sample, prompt_text, generated_text, ref_texts, meta_info in tqdm(zip(samples,
                                                                                  all_prompt_texts,
                                                                                  all_generated_texts,
                                                                                  all_ref_texts, all_meta_infos), desc="Computing metrics"):

            for metric_config in metrics_config_dict:
                # instantiate the config here
                metric = MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
                metric_dict = metric.compute([prompt_text], [generated_text], [ref_texts], [meta_info], model)

                for metric_key, (_, sample_score) in metric_dict.items():
                    metric_values[metric_key].append(sample_score)

    return {metric_name: torch.tensor(metric_values) for metric_name, metric_values in metric_values.items()}


def evaluate_on_samples(model: PreTrainedModel,
                        tokenizer: AutoTokenizer,
                        samples: List[Sample],
                        batch_size: int,
                        max_prompt_length: int,
                        metrics_config_dict: dict,
                        generation_kwargs: dict = None
                        ):
    all_prompt_texts, all_generated_texts, all_ref_texts, all_meta_infos = generate_on_samples(
        model, tokenizer, samples, batch_size, max_prompt_length, generation_kwargs)

    metric_values = __compute_metrics(metrics_config_dict, samples, all_prompt_texts, all_generated_texts, all_ref_texts, all_meta_infos, model)

    return all_generated_texts, metric_values


def __create_output_file_name(config, suffix: str):
    model_name = config["alg"]["model_name"]
    if "/" in model_name:
        model_name = model_name.split("/")[1]

    num_train_samples = config["datapool"].get("num_train_samples", -1)
    train_rnd_seed = config["datapool"].get("train_samples_rnd_seed", -1)

    data_name = config["datapool"]["id"]
    time = datetime.utcnow()
    time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
    return f"train_reward_stats_{data_name}_{model_name}_num_train_{num_train_samples}_rnd_{train_rnd_seed}_{suffix}_{time_str}.pt"


def __save_output(output_dict: dict, config: dict, output_path: str, output_file_suffix: str):
    output_file_name = __create_output_file_name(config, output_file_suffix)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    torch.save(output_dict, os.path.join(output_path, output_file_name))


def main(config: dict, output_path: str, output_file_suffix: str):
    tokenizer_config = config["tokenizer"]
    datapool_config = config["datapool"]
    alg_config = config["alg"]

    tokenizer, model = setup_tokenizer_and_model(tokenizer_config, alg_config)
    train_samples, used_train_indices = get_train_samples(datapool_config)

    eval_config = config["evaluation"]
    gen_kwargs = alg_config["generation_kwargs"]
    max_prompt_length = get_max_prompt_length(tokenizer_config, tokenizer, alg_config, gen_kwargs)

    all_generated_texts, all_metric_values = [], []

    loop_start_time = datetime.utcnow()
    for i in range(eval_config["num_samples_per_input"]):
        start_time = datetime.utcnow()
        print(f"Started computing metrics for generation {i + 1} / {eval_config['num_samples_per_input']}")

        generated_texts, metrics_values = evaluate_on_samples(model,
                                                              tokenizer,
                                                              train_samples,
                                                              eval_config["eval_batch_size"],
                                                              max_prompt_length,
                                                              eval_config["metrics"], gen_kwargs)

        time_delta = datetime.utcnow() - start_time
        print(f"Finished computing metrics for generation {i + 1} / {eval_config['num_samples_per_input']}. Time took: {time_delta}")

        all_generated_texts.append(generated_texts)
        all_metric_values.append(metrics_values)

    print(f"Finished computing metrics. Time took: {datetime.utcnow() - loop_start_time}")

    all_metric_values = {metric_name: torch.stack([metric_values[metric_name] for metric_values in all_metric_values], dim=1).cpu()
                         for metric_name in all_metric_values[0]}
    # Changes format from list of per generation index all sample completions to per sample all completions for that sample
    new_generated_texts = [[generated_texts[i] for generated_texts in all_generated_texts] for i in range(used_train_indices.shape[0])]

    output_dict = {
        "all_generated_texts": new_generated_texts,
        "all_metric_values": all_metric_values,
        "used_train_indices": used_train_indices,
        "num_train_samples": used_train_indices.shape[0],
        "config": config
    }

    __save_output(output_dict, config, output_path, output_file_suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="path to the config file")
    parser.add_argument("--output_path", type=str, default="outputs/rl4lms_stats", help="Base path to store experiment results")
    parser.add_argument("--output_file_suffix", type=str, default="", help="Suffix for output file name")
    args = parser.parse_args()

    # load the config file
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
        main(config, args.output_path, args.output_file_suffix)
