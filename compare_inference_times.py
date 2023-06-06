# Script to compare the inference times of the different models over the test sets of the cities

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_dataset_constructor, get_model
from src.datamodule import ImageAuthorshipDataModule
import time
import pytorch_lightning as pl
from codecarbon import EmissionsTracker
import torch

# Path: compare_inference_times.py

BATCH_SIZES = [1, 10, 100, 1000]
MODELS = ["ELVis", "MF_ELVis", "PRESLEY_8", "PRESLEY_64"]

RESULTS_PATH = "results/"
FIGURES_PATH = "figures/"

WORKERS = 2

PLOT_COLORS = {"ELVis": "g", "PRESLEY_8": "r", "PRESLEY_64": "pink", "MF_ELVis": "b"}

if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

aucs = {}

plt.figure(figsize=(14, 5))
plt.rcParams.update({"font.size": 17})

if __name__ == "__main__":
    for batch_size in BATCH_SIZES:
        for use_cuda in [True, False]:
            for model_name in MODELS:
                if model_name == "ELVis":
                    d = 256
                elif model_name == "MF_ELVis":
                    d = 1024
                elif model_name == "PRESLEY_8":
                    d = 8
                elif model_name == "PRESLEY_64":
                    d = 64

                model = get_model(
                    model_name if model_name[:7] != "PRESLEY" else "PRESLEY",
                    config={"d": d, "lr": 0.01, "dropout": 0.5},
                    nusers=20000,
                )

                # Time the inference and track the emissions
                emissions_tracker = EmissionsTracker(log_level="error")

                random_input = [
                    torch.randint(0, 20000 - 1, (batch_size,)),
                    torch.rand((batch_size, 1536)),
                ]

                if use_cuda:
                    random_input[0] = random_input[0].to("cuda")
                    random_input[1] = random_input[1].to("cuda")
                    model = model.to("cuda")

                emissions_list = []
                times_list = []

                emissions_tracker.start()
                for i in range(1000000):
                    model(random_input)
                emissions_tracker.stop()

                t = emissions_tracker.final_emissions_data.duration
                emissions = np.mean(emissions_tracker.final_emissions * 1000)

                if batch_size not in aucs:
                    aucs[batch_size] = {}
                if str(use_cuda) not in aucs[batch_size]:
                    aucs[batch_size][str(use_cuda)] = {}
                if model_name not in aucs[batch_size][str(use_cuda)]:
                    aucs[batch_size][str(use_cuda)][model_name] = {}

                aucs[batch_size][str(use_cuda)][model_name]["time"] = t
                aucs[batch_size][str(use_cuda)][model_name]["emissions"] = emissions

                print(
                    f"Batch size: {batch_size}, Cuda: {use_cuda}, Model: {model_name}, Time: {t:.2f}, Emissions: {emissions:.2f}"
                )

    # Dump the results
    json.dump(aucs, open(RESULTS_PATH + "inference_times.json", "w"))

    # For each each use_cuda, plot a grouped bar chart (one group per batch size)
    for use_cuda in [True, False]:
        plt.figure(figsize=(14, 5))
        plt.rcParams.update({"font.size": 17})
        plt.title(f"Use CUDA: {use_cuda}")
        plt.xlabel("Batch size")
        plt.ylabel("Time (ms)")
        plt.xticks(np.arange(len(BATCH_SIZES)), BATCH_SIZES)
        for i, model_name in enumerate(MODELS):
            plt.bar(
                np.arange(len(BATCH_SIZES)) + i * 0.2,
                [
                    aucs[batch_size][str(use_cuda)][model_name]["time"]
                    for batch_size in BATCH_SIZES
                ],
                width=0.2,
                color=PLOT_COLORS[model_name],
                label=model_name,
            )
            # Write the values of the bars
            for j, batch_size in enumerate(BATCH_SIZES):
                plt.text(
                    j + i * 0.2 - 0.05,
                    aucs[batch_size][str(use_cuda)][model_name]["time"],
                    f"{aucs[batch_size][str(use_cuda)][model_name]['time']:.2f}",
                )
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + f"inference_times_cuda_{use_cuda}.png")

        # Same for the emissions
        plt.figure(figsize=(14, 5))
        plt.rcParams.update({"font.size": 17})
        plt.title(f"Use CUDA: {use_cuda}")
        plt.xlabel("Batch size")
        plt.ylabel("Emissions (gCO2e)")
        plt.xticks(np.arange(len(BATCH_SIZES)), BATCH_SIZES)
        for i, model_name in enumerate(MODELS):
            plt.bar(
                np.arange(len(BATCH_SIZES)) + i * 0.2,
                [
                    aucs[batch_size][str(use_cuda)][model_name]["emissions"]
                    for batch_size in BATCH_SIZES
                ],
                width=0.2,
                color=PLOT_COLORS[model_name],
                label=model_name,
            )
            # Write the values of the bars
            for j, batch_size in enumerate(BATCH_SIZES):
                plt.text(
                    j + i * 0.2 - 0.05,
                    aucs[batch_size][str(use_cuda)][model_name]["emissions"],
                    f"{aucs[batch_size][str(use_cuda)][model_name]['emissions']:.2f}",
                )
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + f"inference_emissions_cuda_{use_cuda}.png")
