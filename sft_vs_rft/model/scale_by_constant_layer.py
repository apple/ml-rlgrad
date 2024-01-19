
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch.nn as nn


class ScaleByConstant(nn.Module):

    def __init__(self, constant: float):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return self.constant * x
