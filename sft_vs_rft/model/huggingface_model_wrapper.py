
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch.nn as nn

class HuggingFaceModelWrapper(nn.Module):

    def __init__(self, huggingface_model) -> None:
        super().__init__()
        self.huggingface_model = huggingface_model

    def forward(self, input_dict):
        output = self.huggingface_model(**input_dict)
        return output.logits