
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from common.experiment.experiments_plan_runner import ExperimentsPlanRunner
from sft_vs_rft.experiment.sft_vs_rft_controlled_experiment import SupervisedVSReinforcementFinetuningControlledExperiment


def main():
    parser = argparse.ArgumentParser()
    ExperimentsPlanRunner.add_experiments_plan_runner_specific_args(parser)
    args = parser.parse_args()

    experiments_plan_runner = ExperimentsPlanRunner()
    experiment = SupervisedVSReinforcementFinetuningControlledExperiment()
    experiments_plan_runner.run(plan_config_path=args.plan_config_path,
                                experiment=experiment,
                                disable_console_log=args.disable_console_log,
                                save_logs=args.save_logs,
                                log_dir=args.log_dir,
                                log_file_name_prefix=args.log_file_name_prefix)


if __name__ == "__main__":
    main()
