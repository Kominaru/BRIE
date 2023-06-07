#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2023 Iñaki Amatria-Barral
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from statsmodels.stats.libqsturng import qsturng


def _validate_data(data):
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dictionary")

    if len(data) < 3:
        raise ValueError("`data` must have at least 3 keys")

    lenghts = []
    for _, value in data.items():
        if not isinstance(value, list):
            raise TypeError("`data` values must be lists")
        lenghts.append(len(value))

    if len(set(lenghts)) != 1:
        raise ValueError("`data` values must have the same length")


def _compute_cd_lines(avg_data, cd):
    ret = []

    i = 0
    last_line = avg_data[0][1]
    while i < len(avg_data):
        j = i + 1
        while j < len(avg_data):
            if avg_data[i][1] - avg_data[j][1] > cd:
                break
            j += 1
        j -= 1
        if avg_data[j][1] < last_line:
            ret.append((avg_data[i][1], avg_data[j][1]))
            last_line = avg_data[j][1]
        i += 1
    return ret


def _plot_critical_difference(avg_data, cd, n_methods, alpha):
    limits = (n_methods, 1)

    avg_data_list = sorted(list(avg_data.items()), key=lambda x: x[1], reverse=True)

    cd_intervals = _compute_cd_lines(avg_data_list, cd)

    mid = sum(limits) / 2
    right_avg_data = list(filter(lambda x: x[1] < mid, avg_data_list))[::-1]
    left_avg_data = list(filter(lambda x: x[1] >= mid, avg_data_list))

    cd_y = 0.35
    axis_y = cd_y + 0.4
    interval_top_y = axis_y + 0.1
    interval_bot_y = interval_top_y + len(cd_intervals) * 0.1
    labels_top_y = interval_bot_y
    labels_bot_y = labels_top_y + max(len(right_avg_data), len(left_avg_data)) * 0.25

    height = labels_bot_y

    # set up the canvas
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"

    # create the canvas
    _, ax = plt.subplots(figsize=(n_methods + 1, height))
    plt.subplots_adjust(left=0.2, right=0.8, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_title(f"Nemenyi's test critical difference plot ($\\alpha = {alpha}$)")

    # adjust the axis limits
    ax.set_xlim(limits)
    ax.set_ylim((0, height))

    # adjust the axis ticks and position
    ax.xaxis.set_ticks_position("top")
    ax.yaxis.set_visible(False)

    ax.spines["top"].set_position(("data", height - axis_y))
    for p in ["right", "bottom", "left"]:
        ax.spines[p].set_visible(False)

    # plot the critical difference
    cd_pos_corr = 0.02
    cd_left_x = limits[0] - cd_pos_corr
    cd_right_x = limits[0] - cd_pos_corr - cd
    cd_middle_x = cd_left_x - (cd / 2)

    ax.plot([cd_left_x, cd_right_x], [height - cd_y, height - cd_y], c="k", lw=1)
    ax.plot(
        [cd_left_x, cd_left_x],
        [height - cd_y - 0.05, height - cd_y + 0.05],
        c="k",
        lw=1,
    )
    ax.plot(
        [cd_right_x, cd_right_x],
        [height - cd_y - 0.05, height - cd_y + 0.05],
        c="k",
        lw=1,
    )
    ax.text(
        cd_middle_x, height - cd_y + 0.125, f"CD = ${cd:.2f}$", ha="center", va="center"
    )

    # plot the critical intervals
    interval_height = height - interval_top_y
    for left, right in cd_intervals:
        ax.plot([left, right], [interval_height, interval_height], c="k", lw=3)
        interval_height -= 0.1

    # plot the method names
    props = {
        "xycoords": "data",
        "textcoords": "data",
        "va": "center",
        "arrowprops": {
            "arrowstyle": "-",
            "connectionstyle": "angle,angleA=0,angleB=90",
        },
    }

    label_height = height - labels_top_y
    for i, (label, avg_rank) in enumerate(right_avg_data):
        ax.annotate(
            label,
            xy=(avg_rank, height - axis_y),
            xytext=(1, label_height - i * 0.25),
            ha="left",
            **props,
        )
    for i, (label, avg_rank) in enumerate(left_avg_data):
        ax.annotate(
            label,
            xy=(avg_rank, height - axis_y),
            xytext=(n_methods, label_height - i * 0.25),
            ha="right",
            **props,
        )

    plt.savefig("nemenyi.pdf", bbox_inches="tight", dpi=1200)


def nemenyi_plot(data, alpha=0.05):
    _validate_data(data)

    n_methods = len(data)
    n_datasets = len(data[list(data.keys())[0]])

    # compute the friedmanchi square test
    test_stat, p_value = stats.friedmanchisquare(*data.values())
    if not p_value < 0.05:
        warnings.warn(
            "The Friedman test did not reject the null hypothesis"
            f" (test_stat: {test_stat:.4f}, p_value: {p_value:.4f})"
        )

    # compute the critical difference
    q_alpha = qsturng(1 - alpha, len(data), np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt((n_methods * (n_methods + 1)) / (6 * n_datasets))

    # compute the rank matrix
    arr_data = np.array(list(data.values()))
    ranks = np.transpose(stats.rankdata(-arr_data.T, axis=1))
    avg_ranks = np.mean(ranks, axis=1)

    avg_data = {method: avg_rank for method, avg_rank in zip(data.keys(), avg_ranks)}

    # plot the critical difference diagram
    _plot_critical_difference(avg_data, cd, n_methods, alpha)


if __name__ == "__main__":
    # data = {
    #     'MIRLO': [0.912, 0.656, 0.986, 0.906, 0.960],
    #     'RPITER': [0.944, 0.684, 0.991, 0.903, 0.964],
    #     'LION': [0.911, 0.656, 0.994, 0.907, 0.964],
    #     'LncADeep$_{LION}$': [0.900, 0.674, 0.994, 0.898, 0.958],
    #     'rpiCOOL$_{LION}$': [0.921, 0.719, 0.994, 0.904, 0.962],
    #     'RPISeq$_{LION}$': [0.909, 0.683, 0.993, 0.898, 0.964]
    # }
    # data = {
    #     'MIRLO': [0.910, 0.678, 0.984, 0.844, 0.949],
    #     'RPITER': [0.929, 0.724, 0.992, 0.872, 0.947],
    #     'LION': [0.910, 0.716, 0.995, 0.905, 0.952],
    #     'LncADeep$_{LION}$': [0.902, 0.701, 0.995, 0.906, 0.949],
    #     'rpiCOOL$_{LION}$': [0.915, 0.749, 0.995, 0.904, 0.952],
    #     'RPISeq$_{LION}$': [0.908, 0.726, 0.994, 0.909, 0.952]
    # }
    # data = {
    #     'MIRLO': [0.936, 0.802, 0.988, 0.911, 0.853],
    #     'RPITER': [0.949, 0.743, 0.991, 0.916, 0.905],
    #     'LION': [0.929, 0.804, 0.994, 0.910, 0.879],
    #     'LncADeep$_{LION}$': [0.923, 0.820, 0.995, 0.917, 0.857],
    #     'rpiCOOL$_{LION}$': [0.936, 0.828, 0.995, 0.917, 0.851],
    #     'RPISeq$_{LION}$': [0.929, 0.804, 0.995, 0.915, 0.850]
    # }
    data = {
        "RND": [0.487, 0.502, 0.499, 0.502, 0.502, 0.500],
        "ELVis": [0.592, 0.631, 0.638, 0.637, 0.630, 0.629],
        "MF_ELVis": [0.596, 0.596, 0.601, 0.602, 0.596, 0.597],
        "PRESLEY": [0.623, 0.653, 0.662, 0.663, 0.651, 0.654],
    }

    nemenyi_plot(data, alpha=0.1)
