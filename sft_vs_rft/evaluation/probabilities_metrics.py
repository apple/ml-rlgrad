
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch

import common.evaluation.metrics as metrics


class AverageProbabilityDistFromBoundary(metrics.AveragedMetric):

    def _calc_metric(self, y_pred_probs, y):
        return torch.minimum(y_pred_probs, 1 - y_pred_probs).mean().item(), y_pred_probs.shape[0]


class MinCorrectProbability(metrics.MinMetric):

    def _calc_metric(self, y_pred_probs, y):
        return y_pred_probs[torch.arange(y_pred_probs.shape[0]), y].min().item()


class MinProbabilityEntropy(metrics.MinMetric):

    def _calc_metric(self, y_pred_probs, y):
        entropy = torch.special.entr(y_pred_probs).sum(dim=1)
        return entropy.min().item()


class AverageProbabilityEntropy(metrics.AveragedMetric):

    def _calc_metric(self, y_pred_probs, y):
        entropy = torch.special.entr(y_pred_probs).sum(dim=1)
        return entropy.mean().item(), y_pred_probs.shape[0]


class MaxIncorrectProbability(metrics.MaxMetric):

    def _calc_metric(self, y_pred_probs, y):
        clone_y_pred_probs = y_pred_probs.clone()
        clone_y_pred_probs[torch.arange(0, clone_y_pred_probs.shape[0]), y] = -torch.inf
        return clone_y_pred_probs.max().item()


class MaxProbabilityEntropy(metrics.MaxMetric):

    def _calc_metric(self, y_pred_probs, y):
        entropy = torch.special.entr(y_pred_probs).sum(dim=1)
        return entropy.max().item()
