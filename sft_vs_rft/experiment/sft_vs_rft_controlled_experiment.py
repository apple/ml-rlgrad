
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import logging
import math
from typing import Tuple

import numpy as np
import torch.nn as nn
from transformers.models.bert import BertTokenizer, BertForSequenceClassification, BertConfig

import sft_vs_rft.utils.rewards_utils as reward_utils
from common.data.modules import DataModule
from common.evaluation.evaluators import Evaluator, TrainEvaluator, SupervisedValidationEvaluator, SupervisedTrainEvaluator, ComposeEvaluator, \
    ComposeTrainEvaluator, TrainBatchOutputEvaluator
from common.experiment import FitExperimentBase
from common.experiment.fit_experiment_base import ScoreInfo
from common.models.mlp import MultiLayerPerceptron
from common.train.callbacks import Callback
from common.train.trainer import Trainer
from common.utils import model as model_utils
from sft_vs_rft.data.bert_stsb_with_label_modification_datamodule import BertSTSBWithLabelModificationDataModule
from sft_vs_rft.data.torchvision_with_label_modification_datamodule import TorchvisionWithLabelModificationDataModule
from sft_vs_rft.evaluation.probabilities_metrics import *
from sft_vs_rft.model.huggingface_model_wrapper import HuggingFaceModelWrapper
from sft_vs_rft.model.scale_by_constant_layer import ScaleByConstant
from sft_vs_rft.train.reinforce_trainer import ReinforceTrainer
from sft_vs_rft.train.soft_ce_loss import soft_ce_loss, SoftCrossEntropyLossOverProbabilities
from sft_vs_rft.train.supervised_kl_trainer import SupervisedKLTrainer
from sft_vs_rft.utils.rewards_utils import RewardFn


class SupervisedVSReinforcementFinetuningControlledExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(p):
        FitExperimentBase.add_experiment_base_specific_args(p)

        p.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")

        p.add_argument("--num_train_samples", type=int, default=1000, help="Number of training samples")
        p.add_argument("--num_test_samples", type=int, default=1000, help="Number of test samples")
        p.add_argument("--samples_rnd_seed", type=int, default=-1, help="Random seed for modifying labels and adding samples")
        p.add_argument("--num_labels_per_sample", type=int, default=1, help="If > 1, will sample multiple labels per input.")
        p.add_argument("--use_multiple_labels_per_sample", action="store_true",
                       help="If True, will use a cross entropy loss with respect to a uniform "
                            "distribution for this number of labels per sample (the original label "
                            "is one of the labels)")
        p.add_argument("--frac_samples_to_modify_labels", type=float, default=0, help="Fraction of train and test samples to modify their labels")

        p.add_argument("--model_type", typ=str, default="resnet18", help="Model to use. E.g., supports 'mlp', 'resnet18', 'resnet34', "
                                                                         "'resnet50', 'vgg16', 'vgg19'.")
        p.add_argument("--mlp_hidden_layer_sizes", nargs="+", type=int, default=[250, 100], help="List of hidden layer sizes for MLP")
        p.add_argument("--init_rnd_seed", type=int, default=-1, help="Random seed for initializing weights")

        p.add_argument("--opt_method", type=str, default="supervised", help="Optimization method to use. Supports 'supervised' for binary cross "
                                                                            "entropy loss, 'reinforce' for the (sample) based REINFORCE algorithm, "
                                                                            "and 'expected_reinforce' for using expectation of REINFORCE steps (i.e. "
                                                                            "using gradient ascent on the value function).")
        p.add_argument("--reward_type", type=str, default="labels", help="Type of reward function to use for reinforcement learning based "
                                                                         "optimization. Supports 'labels' that gives +1 to correct prediction "
                                                                         "and -1 otherwise, and 'labels_with_custom' that gives +1 to correct prediction, "
                                                                         "-1 to incorrect prediction for sample without modified label and 0.5 to incorrect "
                                                                         "prediction for sample with modified label.")
        p.add_argument("--min_reward", type=float, default=-1, help="Min reward to use for 'labels' reward type.")

        p.add_argument("--optimizer", type=str, default="sgd", help="optimizer to use. Supports: 'sgd' and 'adam'.")
        p.add_argument("--lr", type=float, default=0.001, help="Optimization learning rate")
        p.add_argument("--weight_decay", type=float, default=0, help="Weight decay coefficient")
        p.add_argument("--kl_regularization_coeff", type=float, default=0, help="KL regularization coefficient")
        p.add_argument("--kl_reg_ref_unif", action="store_true", help="Use uniform reference distribution for KL regularization")
        p.add_argument("--logits_temperature", type=float, default=1, help="Temperature applied to logits")
        p.add_argument("--batch_size", type=int, default=128, help="Train/test batch size")

        p.add_argument("--load_model_from_checkpoint", type=str, default="", help="Loads the model from the given checkpoint path.")

    def validate_config(self, config: dict):
        super().validate_config(config)
        if config["use_multiple_labels_per_sample"] and config["num_labels_per_sample"] <= 1:
            raise ValueError("'num_labels_per_sample' must be greater than 1 when 'use_multiple_labels_per_sample' is True")
        if config["use_multiple_labels_per_sample"] and config["opt_method"] != "supervised":
            raise ValueError("'use_multiple_labels_per_sample' is supported only when 'opt_method' is 'supervised'")
        if (config["dataset"] == "stsb") != (config["model_type"].startswith("bert")):
            raise ValueError("The dataset 'stsb' can only be used with bert models")

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        if config["samples_rnd_seed"] >= 0:
            curr_rng_state = torch.random.get_rng_state()
            np_rng_state = np.random.get_state()
            np.random.seed(config["samples_rnd_seed"])
            torch.random.manual_seed(config["samples_rnd_seed"])

        if config["dataset"] != "stsb":
            datamodule = TorchvisionWithLabelModificationDataModule(dataset_name=config["dataset"],
                                                                    num_train_samples=config["num_train_samples"],
                                                                    num_test_samples=config["num_test_samples"],
                                                                    batch_size=config["batch_size"],
                                                                    num_labels_per_sample=config["num_labels_per_sample"],
                                                                    use_multiple_labels_per_sample=config["use_multiple_labels_per_sample"],
                                                                    frac_train_samples_modify_label=config["frac_samples_to_modify_labels"])
        else:
            tokenizer = BertTokenizer.from_pretrained(f"prajjwal1/{config['model_type']}")
            tokenizer.model_max_length = 512
            datamodule = BertSTSBWithLabelModificationDataModule(tokenizer,
                                                                 batch_size=config["batch_size"],
                                                                 num_train_samples=config["num_train_samples"],
                                                                 num_labels_per_sample=config["num_labels_per_sample"],
                                                                 use_multiple_labels_per_sample=config["use_multiple_labels_per_sample"],
                                                                 frac_train_samples_modify_label=config["frac_samples_to_modify_labels"])

        datamodule.setup()

        if config["samples_rnd_seed"] >= 0:
            np.random.set_state(np_rng_state)
            torch.random.set_rng_state(curr_rng_state)

        return datamodule

    def create_model(self, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        if config["init_rnd_seed"] >= 0:
            curr_rng_state = torch.random.get_rng_state()
            torch.random.manual_seed(config["init_rnd_seed"])

        num_classes = datamodule.num_classes

        if config["model_type"] == "mlp":
            model = MultiLayerPerceptron(math.prod(datamodule.input_dims), num_classes, hidden_layer_sizes=config["mlp_hidden_layer_sizes"])
        elif not config["model_type"].startswith("bert"):
            model = model_utils.create_modified_model(config["model_type"], input_size=datamodule.input_dims, output_size=num_classes)
        else:
            bert_config = BertConfig.from_pretrained(f"prajjwal1/{config['model_type']}")
            bert_config.hidden_dropout_prob = 0
            bert_config.attention_probs_dropout_prob = 0
            bert_config.classifier_dropout = 0
            bert_config.num_labels = num_classes
            model = HuggingFaceModelWrapper(BertForSequenceClassification(bert_config))

        if config["init_rnd_seed"] >= 0:
            torch.random.set_rng_state(curr_rng_state)

        model = nn.Sequential(model, ScaleByConstant(constant=1 / config["logits_temperature"]), nn.Softmax(dim=1))
        if config["load_model_from_checkpoint"]:
            checkpoint = torch.load(config["load_model_from_checkpoint"], map_location=state["device"])
            model.load_state_dict(checkpoint["model"])

        return model

    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: DataModule, device, config: dict, state: dict,
                                               logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = self.__get_supervised_metrics("train", config)
        train_evaluators = [SupervisedTrainEvaluator(train_metric_info_seq)]

        batch_output_metrics_to_track = ["reg objective value", "kl reg", "params_norm", "grad norm", "normalized grad norm"]
        if not config["use_multiple_labels_per_sample"]:
            batch_output_metrics_to_track += ["reward"]

        if config["frac_samples_to_modify_labels"] > 0:
            batch_output_metrics_to_track.extend(["grad norm small reward std", "reg objective small reward std",
                                                  "train accuracy small reward std", "normalized grad norm small reward std",
                                                  "avg reward std small reward std", "reward small reward std"])
            if config["frac_samples_to_modify_labels"] < 1:
                batch_output_metrics_to_track.extend(["grad norm large reward std", "reg objective large reward std",
                                                      "train accuracy large reward std", "normalized grad norm large reward std",
                                                      "avg reward std large reward std", "reward large reward std"])

        train_evaluators.append(TrainBatchOutputEvaluator(batch_output_metrics_to_track))

        val_metric_info_seq = self.__get_supervised_metrics("test", config)
        val_evaluators = [SupervisedValidationEvaluator(model, datamodule.test_dataloader(), metric_info_seq=val_metric_info_seq, device=device)]

        return ComposeTrainEvaluator(train_evaluators), ComposeEvaluator(val_evaluators)

    def __get_supervised_metrics(self, phase: str, config: dict):
        if not config["use_multiple_labels_per_sample"]:
            return [
                metrics.MetricInfo(f"{phase} loss", metrics.CrossEntropyLossOverProbabilities(), tag="loss"),
                metrics.MetricInfo(f"{phase} accuracy", metrics.TopKAccuracyWithLogits(k=1), tag="accuracy")
            ]
        else:
            return [
                metrics.MetricInfo(f"{phase} loss", SoftCrossEntropyLossOverProbabilities(), tag="loss"),
            ]

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="train loss", is_train_metric=True, largest=False, return_best_score=False)

    def create_additional_metadata_to_log(self, model: torch.nn.Module, datamodule: DataModule,
                                          config: dict, state: dict, logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        additional_metadata["num train samples"] = len(datamodule.train_dataset)
        additional_metadata["num test samples"] = len(datamodule.test_dataset)
        return additional_metadata

    def create_trainer(self, model: nn.Module, datamodule: DataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        optimizer = self.__create_optimizer(model, config)
        reward_fn = self.__create_reward_function(datamodule, config)

        if config["opt_method"] == "supervised":
            loss_fn = self.__get_supervised_loss_fn(config)
            return SupervisedKLTrainer(model, optimizer, loss_fn, report_grad_norm=True,
                                       reward_fn=reward_fn,
                                       kl_regularization_coeff=config["kl_regularization_coeff"],
                                       kl_reg_ref_unif=config["kl_reg_ref_unif"],
                                       report_metrics_for_modified_sample_separately=config["frac_samples_to_modify_labels"] > 0,
                                       train_evaluator=train_evaluator,
                                       val_evaluator=val_evaluator, callback=callback, device=device)

        elif config["opt_method"] in ["reinforce", "expected_reinforce"]:
            expected_reinforce_updates = config["opt_method"] == "expected_reinforce"
            return ReinforceTrainer(model, optimizer, reward_fn, expected_reinforce_updates=expected_reinforce_updates, report_grad_norm=True,
                                    kl_regularization_coeff=config["kl_regularization_coeff"],
                                    kl_reg_ref_unif=config["kl_reg_ref_unif"],
                                    report_metrics_for_modified_and_unmodified=True,
                                    train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                    callback=callback, device=device)

        else:
            raise ValueError(f"Unsupported optimization method: {config['opt_method']}")

    def __create_optimizer(self, model: nn.Module, config: dict):
        if config["optimizer"] == "adam":
            return torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "sgd":
            return torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

    def __get_supervised_loss_fn(self, config):
        if not config["use_multiple_labels_per_sample"]:
            return lambda y_pred_probs, y: torch.nn.functional.nll_loss(torch.log(y_pred_probs), y)

        return soft_ce_loss

    def __create_reward_function(self, datamodule: DataModule, config: dict) -> RewardFn:
        if config["samples_rnd_seed"] >= 0:
            curr_rng_state = torch.random.get_rng_state()
            torch.random.manual_seed(config["samples_rnd_seed"] + 1)

        if config["reward_type"] == "labels":
            reward_fn = reward_utils.LabelsReward(min_reward=config["min_reward"])
        elif config["reward_type"] == "labels_with_custom":
            reward_fn = reward_utils.LabelsRewardWithCustomRewardForModifiedLabels(min_reward=config["min_reward"],
                                                                                   modified_labels_indicator=datamodule.train_modified_indicator)
        else:
            raise ValueError(f"Unsupperted reward type: {config['reward_type']}")

        if config["samples_rnd_seed"] >= 0:
            torch.random.set_rng_state(curr_rng_state)

        return reward_fn
