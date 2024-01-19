
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import copy

import torch

import common.utils.module as module_utils
from common.evaluation.evaluators.evaluator import VoidEvaluator
from common.train.trainer import Trainer


class SupervisedKLTrainer(Trainer):
    """
    Trainer for supervised classification task, supporting KL regularization.
    """

    def __init__(self, model, optimizer, loss_fn, report_grad_norm: bool = False, reward_fn=None, kl_regularization_coeff: float = 0,
                 load_kl_reg_ref_model_from_checkpoint: str = "", kl_reg_ref_unif: bool = False,
                 report_metrics_for_modified_sample_separately: bool = False, train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(), callback=None,
                 device=torch.device("cpu")):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn
        self.report_grad_norm = report_grad_norm
        self.reward_fn = reward_fn
        self.load_kl_reg_ref_model_from_checkpoint = load_kl_reg_ref_model_from_checkpoint
        self.kl_reg_ref_unif = kl_reg_ref_unif
        self.report_metrics_for_modified_sample_separately = report_metrics_for_modified_sample_separately

        self.kl_regularization_coeff = kl_regularization_coeff
        if self.kl_regularization_coeff != 0 and not self.kl_reg_ref_unif:
            self.init_model_copy = copy.deepcopy(model)

            if self.load_kl_reg_ref_model_from_checkpoint:
                checkpoint = torch.load(self.load_kl_reg_ref_model_from_checkpoint, map_location=device)
                self.init_model_copy.load_state_dict(checkpoint["model"])

            self.init_model_copy.requires_grad_(False)
            self.init_model_copy.to(device)

    def batch_update(self, batch_num, batch, total_num_batches):
        x, y, indices, modified_indicator = batch
        if not isinstance(x, dict):
            x = x.to(self.device)
        else:
            x = {key: value.to(self.device) for key, value in x.items()}

        y = y.to(self.device)

        output = {}
        if self.report_grad_norm:
            output["params_norm"] = module_utils.compute_params_norm(self.model).item()
        if self.report_metrics_for_modified_sample_separately:
            self.__update_output_with_grad_norm_of_modified_and_unmodified(output, x, y, indices, modified_indicator)

        y_pred_probs = self.model(x)
        regularized_objective_value, loss, kl_reg = self.__compute_regularized_objective_value(x, y_pred_probs, y)

        if len(y.shape) == 1 or y.shape[1] == 1:
            reward, _ = self.__compute_expected_reward_no_grad(y_pred_probs, y, indices)
            output["reward"] = reward.item()

        self.optimizer.zero_grad()
        regularized_objective_value.backward()
        self.optimizer.step()

        if self.report_grad_norm:
            grad_norm = module_utils.compute_grad_norm(self.model, detach=True).item()
            output["grad norm"] = grad_norm
            output["normalized grad norm"] = grad_norm / output["params_norm"]

        output.update({
            "loss": loss.item(),
            "reg objective value": regularized_objective_value.item(),
            "kl reg": kl_reg.item() if isinstance(kl_reg, torch.Tensor) else kl_reg,
            "y_pred": y_pred_probs.detach(),
            "y": y,
        })
        return output

    def __compute_regularized_objective_value(self, x: torch.Tensor, y_pred_probs: torch.Tensor, y: torch.Tensor):
        loss = self.loss_fn(y_pred_probs, y)
        kl_reg = self.__expected_kl_reg(x, y_pred_probs).mean() if self.kl_regularization_coeff != 0 else 0
        regularized_objective_value = loss + self.kl_regularization_coeff * kl_reg

        return regularized_objective_value, loss, kl_reg

    def __expected_kl_reg(self, x: torch.tensor, y_pred_probs: torch.Tensor) -> torch.Tensor:
        reference_y_pred_probs = self.init_model_copy(x) if not self.kl_reg_ref_unif else torch.full_like(y_pred_probs,
                                                                                                          fill_value=0.5 if y_pred_probs.shape[1] == 1
                                                                                                          else 1 / y_pred_probs.shape[1])

        if y_pred_probs.shape[1] == 1:
            y_pred_probs = torch.cat([1 - y_pred_probs, y_pred_probs], dim=1)
            reference_y_pred_probs = torch.cat([1 - reference_y_pred_probs, reference_y_pred_probs], dim=1)

        kl_div_entries = y_pred_probs * torch.log(y_pred_probs / reference_y_pred_probs)
        return kl_div_entries.sum(dim=1)

    def __update_output_with_grad_norm_of_modified_and_unmodified(self, output, x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor,
                                                                 modified_indicator: torch.Tensor):
        if modified_indicator.any():
            if not isinstance(x, dict):
                x_modified = x[modified_indicator]
            else:
                x_modified = {key: value[modified_indicator] for key, value in x.items()}

            y_pred_probs_modified = self.model(x_modified)
            reg_objective_value_modified, _, _ = self.__compute_regularized_objective_value(x_modified,
                                                                                           y_pred_probs_modified,
                                                                                           y[modified_indicator])
            self.optimizer.zero_grad()
            (- reg_objective_value_modified).backward()

            grad_norm = module_utils.compute_grad_norm(self.model, detach=True).item()
            output["grad norm small reward std"] = grad_norm
            output["normalized grad norm small reward std"] = grad_norm / output["params_norm"]
            output["reg objective small reward std"] = reg_objective_value_modified.item()

            reward_modified, all_rewards_modified = self.__compute_expected_reward_no_grad(y_pred_probs_modified,
                                                                                         y[modified_indicator],
                                                                                         indices[modified_indicator])
            output["reward small reward std"] = reward_modified.item()
            output["avg reward std small reward std"] = self.__compute_reward_std(y_pred_probs_modified, all_rewards_modified).item()

            if len(y.shape) == 1 or y.shape[1] == 1:
                train_acc_modified = (y_pred_probs_modified.argmax(dim=1) == y[modified_indicator]).to(torch.float).mean()
                output["train accuracy small reward std"] = train_acc_modified.item()

        if (~modified_indicator).any():
            if not isinstance(x, dict):
                x_unmodified = x[~modified_indicator]
            else:
                x_unmodified = {key: value[~modified_indicator] for key, value in x.items()}

            y_pred_probs_unmodified = self.model(x_unmodified)
            reg_objective_value_unmodified, _, _ = self.__compute_regularized_objective_value(x_unmodified,
                                                                                              y_pred_probs_unmodified,
                                                                                              y[~modified_indicator])
            self.optimizer.zero_grad()
            (- reg_objective_value_unmodified).backward()

            grad_norm = module_utils.compute_grad_norm(self.model, detach=True).item()
            output["grad norm large reward std"] = grad_norm
            output["normalized grad norm large reward std"] = grad_norm / output["params_norm"]
            output["reg objective large reward std"] = reg_objective_value_unmodified.item()

            reward_unmodified, all_rewards_unmodified = self.__compute_expected_reward_no_grad(y_pred_probs_unmodified,
                                                                                               y[~modified_indicator],
                                                                                               indices[~modified_indicator])
            output["reward large reward std"] = reward_unmodified.item()
            output["avg reward std large reward std"] = self.__compute_reward_std(y_pred_probs_unmodified, all_rewards_unmodified).item()

            if len(y.shape) == 1 or y.shape[1] == 1:
                train_acc_unmodified = (y_pred_probs_unmodified.argmax(dim=1) == y[~modified_indicator]).to(torch.float).mean()
                output["train accuracy large reward std"] = train_acc_unmodified.item()

        self.optimizer.zero_grad()

    def __compute_expected_reward_no_grad(self, y_pred_probs: torch.Tensor, y: torch.Tensor, indices: torch.Tensor):
        with torch.no_grad():
            all_rewards = self.__compute_all_rewards(y, indices, y_pred_probs)
            return self.__expected_reinforce_objective(y_pred_probs, all_rewards), all_rewards

    def __expected_reinforce_objective(self, y_pred_probs: torch.Tensor, all_rewards: torch.Tensor) -> torch.Tensor:
        if y_pred_probs.shape[1] > 1:
            expected_reward = (all_rewards * y_pred_probs).sum(dim=1)
            return expected_reward.mean()
        else:
            expected_reward = all_rewards[:, 0] * (1 - y_pred_probs).squeeze(dim=1) + all_rewards[:, 1] * y_pred_probs.squeeze(dim=1)
            return expected_reward.mean()

    def __compute_all_rewards(self, y: torch.Tensor, indices: torch.Tensor, y_pred_probs: torch.Tensor) -> torch.Tensor:
        if y_pred_probs.shape[1] > 1:
            per_label_rewards = []
            for label in range(y_pred_probs.shape[1]):
                per_label_rewards.append(self.reward_fn(torch.full_like(y, fill_value=label), y, indices))

            per_label_rewards = torch.stack(per_label_rewards, dim=1)
            return per_label_rewards
        else:
            zero_label_reward = self.reward_fn(torch.zeros_like(y_pred_probs), y, indices)
            one_label_reward = self.reward_fn(torch.ones_like(y_pred_probs), y, indices)
            return torch.cat([zero_label_reward, one_label_reward], dim=1)

    def __compute_reward_std(self, y_pred_probs: torch.Tensor, all_rewards: torch.Tensor) -> torch.Tensor:
        if y_pred_probs.shape[1] == 1:
            y_pred_probs = torch.cat([1 - y_pred_probs, y_pred_probs], dim=1)

        expected_rewards = (y_pred_probs * all_rewards).sum(dim=1, keepdims=True)
        return torch.sqrt((y_pred_probs * torch.pow(all_rewards - expected_rewards, exponent=2)).sum(dim=1)).mean()
