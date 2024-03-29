
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional, Dict, Any
import torch


class KLController:
    def __init__(self, kl_coeff: float, target_kl: Optional[float] = None) -> None:
        self._kl_coeff = kl_coeff
        self._target_kl = target_kl

    def step(self, kl_div: torch.tensor):
        """
        Adapts the KL coeff
        """
        if self._target_kl is not None:
            diff_to_target = (kl_div - self._target_kl) / self._target_kl
            e_t = torch.clip(diff_to_target, -0.2, 0.2).item()
            self._kl_coeff = self._kl_coeff * (1 + 0.1 * e_t)

    @property
    def kl_coeff(self):
        return self._kl_coeff

    def get_state_dict(self) -> Dict[str, Any]:
        state = {
            "target_kl": self._target_kl,
            "current_kl_coeff": self._kl_coeff
        }
        return state

    def load_from_state_dict(self, state_dict: Dict[str, Any]):
        self._kl_coeff = state_dict["current_kl_coeff"]
        self._target_kl = state_dict["target_kl"]


if __name__ == "__main__":
    contr = KLController(kl_coeff=0.1, target_kl=0.1)

    contr.step(torch.tensor(-0.2))
    print(contr.kl_coeff)

    contr.step(torch.tensor(0.3))
    print(contr.kl_coeff)

    contr.step(torch.tensor(0.4))
    print(contr.kl_coeff)

    state_dict = contr.get_state_dict()
    print(state_dict)

    contr._target_kl = None
    contr._kl_coeff = None
    contr.load_from_state_dict(state_dict)
    assert contr._target_kl == state_dict["target_kl"]
    assert contr._kl_coeff == state_dict["current_kl_coeff"]
