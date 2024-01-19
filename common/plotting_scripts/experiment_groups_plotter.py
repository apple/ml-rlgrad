
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TITLE_SIZE = 13
LABELS_SIZE = 12
TICKS_SIZE = 9
LEGEND_FONT_SIZE = 10
COLORS = ["tab:green", "tab:blue", "tab:purple", "tab:red", "tab:brown", "tab:cyan", "tab:gray", "tab:orange", "tab:olive", "tab:pink"]
MARKERS = ["d", "o", "^", "v", "X", "s", "h", "1", "<", "*"]
DEFAULT_LINESTYLE = "-"
DEFAULT_HORIZONTAL_LINE_COLOR = "black"
DEFAULT_HORIZONTAL_LINE_STYLE = "-"
DEFAULT_VERTICAL_LINE_COLOR = "black"
DEFAULT_VERTICAL_LINE_STYLE = "-"


class ExperimentGroupPlotInfo:

    def __init__(self, dir_name: str, label: str):
        self.dir_name: str = dir_name
        self.label = label

        self.x_value_to_y_values = defaultdict(list)


def __load_summary_json(experiment_dir_path: Path):
    summary_json_path = experiment_dir_path.joinpath("summary.json")

    if not summary_json_path.exists():
        return None

    with open(summary_json_path) as f:
        return json.load(f)


def __load_config_json(experiment_dir_path: Path):
    config_json_path = experiment_dir_path.joinpath("config.json")

    if not config_json_path.exists():
        return None

    with open(config_json_path) as f:
        return json.load(f)


def __extract_value_from_json_dict(summary: dict, key: str):
    key_parts = key.split(".")

    curr_value = summary
    for key_part in key_parts:
        if key_part not in curr_value:
            return None

        curr_value = curr_value[key_part]

    return curr_value


def __create_experiment_group_plot_infos(args):
    experiment_group_plot_infos = []

    for i, experiment_group_dir_name in enumerate(args.experiment_groups_dir_names):
        exp_group_label = args.per_experiment_group_label[i] if args.per_experiment_group_label else ""
        exp_group_plot_info = ExperimentGroupPlotInfo(experiment_group_dir_name, exp_group_label)

        experiment_group_dir_path = Path(os.path.join(args.experiments_dir, experiment_group_dir_name))
        experiment_dir_paths = [path for path in experiment_group_dir_path.iterdir() if path.is_dir()]

        for experiment_dir_path in experiment_dir_paths:
            exp_summary = __load_summary_json(experiment_dir_path)

            if not exp_summary:
                continue

            x_value = __extract_value_from_json_dict(exp_summary, args.x_axis_value_name)
            if x_value is None:
                exp_config = __load_config_json(experiment_dir_path)
                x_value = __extract_value_from_json_dict(exp_config, args.x_axis_value_name)

            y_value = __extract_value_from_json_dict(exp_summary, args.per_experiment_group_y_axis_value_name[i])

            exp_group_plot_info.x_value_to_y_values[x_value].append(y_value * args.y_value_scale_factor)

        experiment_group_plot_infos.append(exp_group_plot_info)

    return experiment_group_plot_infos


def __populate_plot(ax, experiment_group_plot_infos, args):
    linestyles = args.per_experiment_group_linestyle if args.per_experiment_group_linestyle \
        else [DEFAULT_LINESTYLE] * len(experiment_group_plot_infos)
    markers = args.per_experiment_group_marker if args.per_experiment_group_marker else MARKERS
    colors = args.per_experiment_group_color if args.per_experiment_group_color else COLORS

    for i, exp_group_plot_info in enumerate(experiment_group_plot_infos):
        x_values = np.array(list(exp_group_plot_info.x_value_to_y_values.keys()))
        per_x_y_values = [exp_group_plot_info.x_value_to_y_values[x_value] for x_value in x_values]

        per_x_mean_y_values = [np.mean(y_values) for y_values in per_x_y_values]
        y_errs = [np.std(y_values) for y_values in per_x_y_values]

        sorted_order = np.argsort(x_values)
        sorted_x_values = x_values[sorted_order]
        sorted_per_x_mean_y_values = np.array(per_x_mean_y_values)[sorted_order]
        sorted_y_errs = np.array(y_errs)[sorted_order]

        _, _, bars = ax.errorbar(sorted_x_values, sorted_per_x_mean_y_values, yerr=sorted_y_errs, marker=markers[i % len(markers)],
                                 color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=args.plot_linewidth,
                                 label=exp_group_plot_info.label)

        for bar in bars:
            bar.set_alpha(args.error_bars_opacity)

    for i, horizontal_line in enumerate(args.horizontal_lines):
        horizontal_color = args.horizontal_lines_colors[i] if len(args.horizontal_lines_colors) > i else DEFAULT_HORIZONTAL_LINE_COLOR
        horizontal_line_style = args.horizontal_lines_styles[i] if len(args.horizontal_lines_styles) > i else DEFAULT_HORIZONTAL_LINE_STYLE
        ax.axhline(horizontal_line, linewidth=args.plot_linewidth, color=horizontal_color, linestyle=horizontal_line_style)

    for i, vertical_line in enumerate(args.vertical_lines):
        vertical_color = args.vertical_lines_colors[i] if len(args.vertical_lines_colors) > i else DEFAULT_VERTICAL_LINE_COLOR
        vertical_line_style = args.vertical_lines_styles[i] if len(args.vertical_lines_styles) > i else DEFAULT_VERTICAL_LINE_STYLE
        ax.axvline(vertical_line, linewidth=args.plot_linewidth, color=vertical_color, linestyle=vertical_line_style)

    if args.y_log_scale:
        ax.set_yscale("log")

    if args.x_log_scale:
        ax.set_xscale("log")

    ax.set_title(args.plot_title, fontsize=TITLE_SIZE)
    ax.set_ylabel(args.y_label, fontsize=LABELS_SIZE)
    ax.set_xlabel(args.x_label, fontsize=LABELS_SIZE)

    if args.x_autoscale_tight:
        ax.autoscale(enable=True, axis='x', tight=True)

    ax.set_ylim(bottom=args.y_bottom_lim, top=args.y_top_lim)

    ax.tick_params(labelsize=TICKS_SIZE)
    ax.legend(prop={"size": LEGEND_FONT_SIZE}, title=args.legend_title)

    if args.x_axis_ticks:
        ax.set_xticks(args.x_axis_ticks)

    if args.y_axis_ticks:
        ax.set_yticks(args.y_axis_ticks)


def __create_plot_from_experiment_group_plot_infos(experiment_group_plot_infos, args):
    fig, ax = plt.subplots(1, figsize=(args.fig_width, args.fig_height))

    __populate_plot(ax, experiment_group_plot_infos, args)

    plt.tight_layout()
    if args.save_plot_to:
        plt.savefig(args.save_plot_to, dpi=250, bbox_inches='tight', pad_inches=0.1)

    plt.show()


def create_plot(args):
    experiment_group_plot_infos = __create_experiment_group_plot_infos(args)
    __create_plot_from_experiment_group_plot_infos(experiment_group_plot_infos, args)


def __verify_same_of_num_values(first_list, second_list, first_name: str, second_name: str):
    if len(first_list) != len(second_list):
        raise ValueError(f"Mismatch in number values given in '{first_name}' and '{second_name}'. "
                         f"Received lengths are {len(first_list)} and {len(second_list)}.")


def __verify_args(args):
    __verify_same_of_num_values(args.experiment_groups_dir_names, args.per_experiment_group_y_axis_value_name,
                                "experiment_group_dir_names", "per_experiment_y_axis_value_name")

    if len(args.per_experiment_group_label) > 0:
        __verify_same_of_num_values(args.experiment_groups_dir_names, args.per_experiment_group_label,
                                    "experiment_group_dir_names", "per_experiment_group_label")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments_dir", type=str, required=True, help="Path to the directory under which the directories for the "
                                                                      "different experiment groups lie.")
    p.add_argument("--experiment_groups_dir_names", nargs="+", type=str, required=True,
                   help="Name of directories of the experiment groups. If none given, will take experiments under experiments_dir as a single group."
                        "Each experiment in the group directory must have a 'summary.json' file in its own directory, from which the values of "
                        "the plot are taken.")
    p.add_argument("--per_experiment_group_y_axis_value_name", nargs="+", type=str, required=True,
                   help="Name of the value to report for each experiment. Name should match key in experiment's summary.json files."
                        " Use dot notation for nested keys.")
    p.add_argument("--per_experiment_group_label", nargs="+", type=str, default=[],
                   help="Label for each experiment group. Number of values must match the number of experiment_group_dir_names given.")
    p.add_argument("--per_experiment_group_marker", nargs="+", type=str, default=[],
                   help="Marker for each experiment group. Number of values must match the number of experiment_group_dir_names given.")
    p.add_argument("--per_experiment_group_linestyle", nargs="+", type=str, default=[],
                   help="Linestyle for each experiment group. Number of values must match the number of experiment_group_dir_names given.")
    p.add_argument("--per_experiment_group_color", nargs="+", type=str, default=[],
                   help="Color for each experiment group. Number of values must match the number of experiment_group_dir_names given.")
    p.add_argument("--error_bars_opacity", type=float, default=1, help="Alpha value for the opacity of error bars.")

    p.add_argument("--x_axis_value_name", type=str, required=True,
                   help="Name of the value to use as the x axis. Name should match key in experiment's "
                        "summary.json or config.json files. Use dot notation for nested keys.")

    p.add_argument("--horizontal_lines", nargs="+", type=float, default=[], help="Adds horizontal lines at these y value.")
    p.add_argument("--horizontal_lines_colors", nargs="+", type=str, default=[], help="Colors of horizontal lines (default is black).")
    p.add_argument("--horizontal_lines_styles", nargs="+", type=str, default=[], help="Line styles of horizontal lines (default is solid line).")
    p.add_argument("--vertical_lines", nargs="+", type=float, default=[], help="Adds horizontal lines at these x value.")
    p.add_argument("--vertical_lines_colors", nargs="+", type=str, default=[], help="Colors of vertical lines (default is black).")
    p.add_argument("--vertical_lines_styles", nargs="+", type=str, default=[], help="Line styles of vertical lines (default is solid line).")

    p.add_argument("--y_value_scale_factor", type=float, default=1., help="Scale y values by this multiplicative factor.")
    p.add_argument("--plot_title", type=str, default="", help="Title for the plot.")
    p.add_argument("--legend_title", type=str, default="", help="Title for the plot legend.")
    p.add_argument("--x_label", type=str, default="", help="Label of the x axis.")
    p.add_argument("--x_log_scale", action="store_true", help="Use log scale for x axis.")
    p.add_argument("--x_autoscale_tight", action="store_true", help="Sets edges of plot right at minimal and maximal x values.")
    p.add_argument("--y_label", type=str, default="", help="Label of the y axis.")
    p.add_argument("--y_bottom_lim", type=float, default=None, help="Bottom limit for y axis.")
    p.add_argument("--y_top_lim", type=float, default=None, help="Top limit for y axis.")
    p.add_argument("--y_log_scale", action="store_true", help="Use log scale for y axis.")
    p.add_argument("--x_axis_ticks", type=float, nargs="+", default=None, help="List of values to be placed as x ticks.")
    p.add_argument("--y_axis_ticks", type=float, nargs="+", default=None, help="List of values to be placed as y ticks.")

    p.add_argument("--plot_linewidth", type=float, default=1.5, help="Plots line width.")
    p.add_argument("--fig_width", type=float, default=4, help="Figure width.")
    p.add_argument("--fig_height", type=float, default=3, help="Figure height.")
    p.add_argument("--save_plot_to", type=str, default="", help="Save plot to the given file path (doesn't save if non given).")
    args = p.parse_args()

    __verify_args(args)

    create_plot(args)


if __name__ == "__main__":
    main()
