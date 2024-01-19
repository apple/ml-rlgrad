
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class RewardFn(ABC):

    @abstractmethod
    def get_min_and_max_rewards(self) -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, y_batch_pred: torch.Tensor, y_batch: torch.Tensor, indices_batch: torch.Tensor):
        raise NotImplementedError


class LabelsReward(RewardFn):

    def __init__(self, max_reward: float = 1, min_reward: float = -1):
        self.max_reward = max_reward
        self.min_reward = min_reward

    def __call__(self, y_batch_pred: torch.Tensor, y_batch: torch.Tensor, indices_batch: torch.Tensor) -> torch.Tensor:
        if len(y_batch.shape) == 2:
            y_batch = y_batch.squeeze(dim=1)
        if len(y_batch_pred.shape) == 2:
            y_batch_pred = y_batch_pred.squeeze(dim=1)

        rewards = torch.zeros_like(y_batch, dtype=torch.float)
        correct_preds_indicator = y_batch_pred == y_batch
        rewards[correct_preds_indicator] = self.max_reward
        rewards[~correct_preds_indicator] = self.min_reward
        return rewards

    def get_min_and_max_rewards(self) -> Tuple[float, float]:
        return self.min_reward, self.max_reward


class LabelsRewardWithCustomRewardForModifiedLabels(RewardFn):

    def __init__(self, modified_labels_indicator: torch.Tensor, max_reward: float = 1, min_reward: float = -1, modified_label_incorrect_reward: float = 0.5):
        self.modified_labels_indicator = modified_labels_indicator
        self.max_reward = max_reward
        self.min_reward = min_reward
        self.modified_label_incorrect_reward = modified_label_incorrect_reward

    def __call__(self, y_batch_pred: torch.Tensor, y_batch: torch.Tensor, indices_batch: torch.Tensor) -> torch.Tensor:
        if len(y_batch.shape) == 2:
            y_batch = y_batch.squeeze(dim=1)
        if len(y_batch_pred.shape) == 2:
            y_batch_pred = y_batch_pred.squeeze(dim=1)

        rewards = torch.zeros_like(y_batch, dtype=torch.float)
        correct_preds_indicator = y_batch_pred == y_batch
        rewards[~correct_preds_indicator] = self.min_reward
        rewards[self.modified_labels_indicator[indices_batch]] = self.modified_label_incorrect_reward
        rewards[correct_preds_indicator] = self.max_reward
        return rewards

    def get_min_and_max_rewards(self) -> Tuple[float, float]:
        return self.min_reward, self.max_reward
