
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

class TorchSerializable:

    def state_dict(self) -> dict:
        """
        Returns the state of object as a dict for serialization
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict):
        """
        Loads the object state from the given state dictionary.
        """
        raise NotImplementedError
