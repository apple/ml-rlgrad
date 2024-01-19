
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from rl4lms.algorithms.common.maskable.policies import (
    MaskableActorCriticCnnPolicy,
    MaskableActorCriticPolicy,
    MaskableMultiInputActorCriticPolicy,
)

MlpPolicy = MaskableActorCriticPolicy
CnnPolicy = MaskableActorCriticCnnPolicy
MultiInputPolicy = MaskableMultiInputActorCriticPolicy
