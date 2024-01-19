
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import copy
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List

import jsonlines
import pandas as pd
import wandb
from rich.logging import RichHandler
from transformers import AutoModel


class Tracker:
    def __init__(self,
                 base_path_to_store_results: str,
                 run_config: Dict[str, Any],
                 project_name: str,
                 experiment_name: str,
                 entity_name: str = None,
                 wandb_log: bool = False,
                 log_entire_predictions: bool = False,
                 upload_model_to_wandb: bool = False,
                 log_level: int = logging.DEBUG):
        self._log_level = log_level
        self._base_path_to_store_results = base_path_to_store_results
        self._config = run_config
        self._experiment_name = experiment_name
        self._experiment_name_with_time_stamp = self._experiment_name + "_" + datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
        self._project_name = project_name
        self._entity_name = entity_name
        self.wandb_log = wandb_log
        self._log_entire_predictions = log_entire_predictions
        self._upload_model_to_wandb = upload_model_to_wandb
        self._init()

    def _init(self):
        # create a folder
        self._run_path = os.path.join(self._base_path_to_store_results, self._experiment_name_with_time_stamp)
        os.makedirs(self._run_path, exist_ok=True)

        # store also the config into it
        config_path = os.path.join(self._run_path, f"{self._experiment_name}_config.json")
        with open(config_path, "w") as fp:
            json.dump(self._config, fp)

        # init logger
        log_path = os.path.join(self._run_path, f"{self._experiment_name}_log.txt")
        logging.basicConfig(
            level=self._log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                RichHandler()
            ]
        )

        # init wandb
        if self.wandb_log:
            self._wandb_run = wandb.init(
                entity=self._entity_name,
                project=self._project_name,
                name=self._experiment_name_with_time_stamp,
                config=self._config,
                reinit=True
            )

    def log_predictions(self, epoch: int,
                        split_name: str,
                        predictions: List[Dict]):
        if not self._log_entire_predictions:
            return

        # log them per epoch in a separate file as they can get huge
        prediction_file_at_epoch = os.path.join(self._run_path, f"{self._experiment_name}_epoch_{epoch}_{split_name}_split_predictions.json")
        with open(prediction_file_at_epoch, "w") as fp:
            json.dump(predictions, fp)

        # randomly display few predictions for logging
        predictions_ = copy.deepcopy(predictions)
        random.shuffle(predictions_)
        logging.info(f"Split {split_name} predictions")
        for pred in predictions_[:10]:
            logging.info(pred)

        # for wandb logging, we create a table consisting of predictions
        # we can create one table per split per epoch
        if self.wandb_log and len(predictions) > 0:

            def to_df(predictions):
                columns = predictions[0].keys()
                data_by_column = defaultdict(list)
                for item in predictions:
                    for column in columns:
                        data_by_column[column].append(item.get(column, ""))
                data_df = pd.DataFrame(data_by_column)
                return data_df

            predictions_as_df = to_df(predictions)
            predictions_table_at_epoch = wandb.Table(data=predictions_as_df)
            self._wandb_run.log({f"{split_name}_predictions_at_epoch_{epoch}": predictions_table_at_epoch})

    def log_metrics(self, epoch: int,
                    split_name: str,
                    metrics_dict: Dict[str, float]):
        # for each split, one file
        metric_file_per_split = os.path.join(self._run_path, f"{self._experiment_name}_{split_name}_split_metrics.jsonl")
        metrics_dict_ = {
            "epoch": epoch,
            "metrics": metrics_dict
        }
        with jsonlines.open(metric_file_per_split, "a") as writer:
            writer.write(metrics_dict_)

        # log to wandb
        if self.wandb_log:
            metric_dict_ = {f"{split_name}/{metric_key}": value for metric_key, value in metrics_dict.items()}
            metric_dict_["epoch"] = epoch
            wandb.log(metric_dict_)
            wandb.save(metric_file_per_split)

        # logger
        logging.info(f"{split_name} metrics: {metrics_dict_}")

    def log_rollout_infos(self, rollout_info: Dict[str, float]):
        logging.info(f"Rollout Info: {rollout_info}")
        rollout_info_file = os.path.join(self._run_path, f"{self._experiment_name}_rollout_info.jsonl")
        with jsonlines.open(rollout_info_file, mode="a") as writer:
            writer.write(rollout_info)

        # log to wandb
        if self.wandb_log:
            wandb.log(rollout_info)

    def log_training_infos(self, training_info: Dict[str, float]):
        logging.info(f"Training Info: {training_info}")
        training_info_file = os.path.join(self._run_path, f"{self._experiment_name}_training_info.jsonl")
        with jsonlines.open(training_info_file, mode="a") as writer:
            writer.write(training_info)

        # log to wandb
        if self.wandb_log:
            wandb.log(training_info)

    def done(self):
        if self.wandb_log:
            wandb.finish()

    def save_auto_model(self, model: AutoModel):
        model_dir_path = os.path.join(self._run_path, f"{self._experiment_name}_model")
        model.save_pretrained(model_dir_path)

        if self._upload_model_to_wandb:
            print("Uploading model to WandB")
            shutil.make_archive(model_dir_path, "zip", model_dir_path)
            wandb.save(model_dir_path + ".zip")

    @property
    def checkpoint_base_path(self):
        return os.path.join(self._run_path, f"{self._experiment_name}_checkpoints")

    def log_info(self, msg: str):
        logging.info(msg)


if __name__ == "__main__":
    base_path = "/scratch/test_logs"
    run_config = {
        "param_1": 1,
        "param_2": 2
    }
    predictions = {
        "1": [{"sample_id": "1", "prompt_text": "Hello", "gen_text": "I am there"},
              {"sample_id": "2", "prompt_text": "Hi", "gen_text": "there"}],
        "2": [{"sample_id": "1", "prompt_text": "Hello", "gen_text": "I am there"},
              {"sample_id": "2", "prompt_text": "Hi", "gen_text": "there"}],
        "3": [{"sample_id": "1", "prompt_text": "Hello", "gen_text": "I am there"},
              {"sample_id": "2", "prompt_text": "Hi", "gen_text": "there"}],
    }

    metrics = {
        "1": {"metric_1": 0.05, "metric_2": 0.1},
        "2": {"metric_1": 0.06, "metric_2": 0.2},
        "3": {"metric_1": 0.06, "metric_2": 0.3},
    }

    rollout_infos = [
        {"ep_len": 2, "ep_reward": 0.4},
        {"ep_len": 3, "ep_reward": 0.5},
        {"ep_len": 3, "ep_reward": 0.5},
    ]

    tracker = Tracker(base_path, run_config, "Test run", True)
    tracker.log_predictions(1, "val", predictions["1"])
    tracker.log_metrics(1, "val", metrics["1"])
    tracker.log_predictions(2, "val", predictions["2"])
    tracker.log_metrics(2, "val", metrics["2"])
    tracker.log_predictions(3, "val", predictions["3"])
    tracker.log_metrics(3, "val", metrics["3"])
    tracker.log_rollout_infos(rollout_infos[0])
    tracker.log_rollout_infos(rollout_infos[1])
    tracker.log_rollout_infos(rollout_infos[2])
    tracker.done()
