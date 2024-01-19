
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

class StopFitIteration(Exception):
    """
    Exception that can be thrown to cause the fitting process of the trainer to gracefully exit.
    """
    pass
