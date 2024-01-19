
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch

import common.evaluation.metrics as metrics


def soft_ce_loss(y_pred_probs: torch.Tensor, y: torch.Tensor):
    log_probs = torch.log(y_pred_probs)
    log_probs = torch.gather(log_probs, dim=1, index=y)
    return - log_probs.mean()


class SoftCrossEntropyLossOverProbabilities(metrics.AveragedMetric):
    """
    Soft cross entropy loss metric. Receives as input the probabilities of the prediction and the true labels.
    """

    def __init__(self):
        super().__init__()

    def _calc_metric(self, y_pred_probs, y):
        """
        Calculates the cross entropy loss.
        :param y_pred_probs: probabilities of the predictions.
        :param y: true labels.
        :return: (Soft cross entropy loss value, num samples in input)
        """
        return soft_ce_loss(y_pred_probs, y).item(), y.shape[0]
