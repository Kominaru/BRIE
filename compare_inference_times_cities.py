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
import pickle

# Path: compare_inference_times.py

CITIES = ["gijon", "barcelona", "madrid", "newyork", "paris", "london"]
MODELS = ["ELVis", "MF_ELVis", "PRESLEY_8", "PRESLEY_64", "PRESLEY_256", "PRESLEY_1024"]

RESULTS_PATH = "results/"
FIGURES_PATH = "figures/"

WORKERS = 1

PLOT_COLORS = {"ELVis": "g", "PRESLEY_8": "r", "MF_ELVis": "b"}

RUNS_PER_MODEL = 50

if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

aucs = {}

plt.figure(figsize=(30, 5))
plt.rcParams.update({"font.size": 17})

if __name__ == "__main__":
    for city in CITIES:
        for use_cuda in [True]:
            for model_name in MODELS:
                if model_name == "ELVis":
                    d = 256
                elif model_name == "MF_ELVis":
                    d = 1024
                elif "PRESLEY" in model_name:
                    d = int(model_name.split("_")[1])

                # Time the inference and track the emissions
                emissions_tracker = EmissionsTracker(log_level="error")

                test_cases = pickle.load(
                    open("data/" + city + "/data_10+10/TEST_IMG", "rb")
                )
                test_cases_sizes = test_cases.groupby("id_test").size().values
                num_of_unique_users = len(test_cases["id_user"].unique())

                # Split the test cases into batches with at most 1e6 images

                test_cases_batches = []
                current_batch = []
                current_batch_size = 0

                random_users = torch.randint(
                    0, num_of_unique_users - 1, (max(test_cases_sizes),)
                ).to("cuda" if use_cuda else "cpu")

                random_images = torch.rand((max(test_cases_sizes), 1536)).to(
                    "cuda" if use_cuda else "cpu"
                )

                model = get_model(
                    model_name if model_name[:7] != "PRESLEY" else "PRESLEY",
                    config={"d": d, "lr": 0.01, "dropout": 0.5},
                    nusers=num_of_unique_users,
                )

                model = model.to("cuda" if use_cuda else "cpu")

                emissions_tracker = EmissionsTracker(log_level="error")

                emissions_tracker.start()

                for _ in range(RUNS_PER_MODEL):
                    for test_case_size in test_cases_sizes:
                        model(
                            (
                                random_users[:test_case_size],
                                random_images[:test_case_size],
                            )
                        )

                emissions_tracker.stop()

                t = emissions_tracker.final_emissions_data.duration / RUNS_PER_MODEL
                emissions = emissions_tracker.final_emissions * 1000 / RUNS_PER_MODEL

                if city not in aucs:
                    aucs[city] = {}
                if str(use_cuda) not in aucs[city]:
                    aucs[city][str(use_cuda)] = {}
                if model_name not in aucs[city][str(use_cuda)]:
                    aucs[city][str(use_cuda)][model_name] = {}

                aucs[city][str(use_cuda)][model_name]["time"] = t
                aucs[city][str(use_cuda)][model_name]["emissions"] = emissions

                print(
                    f"Batch size: {city}, Cuda: {use_cuda}, Model: {model_name}, Time: {t:.2f}, Emissions: {emissions:.2f}"
                )

    # Dump the results
    json.dump(aucs, open(RESULTS_PATH + "inference_times.json", "w"))

    width = 0.15
    # For each each use_cuda, plot a grouped bar chart (one group per batch size)
    for use_cuda in [True]:
        plt.figure(figsize=(16, 7))
        plt.rcParams.update({"font.size": 17})
        plt.title(f"Use CUDA: {use_cuda}")
        plt.xlabel("Batch size")
        plt.ylabel("Time (ms)")
        plt.xticks(np.arange(len(CITIES)), CITIES)
        for i, model_name in enumerate(MODELS):
            plt.bar(
                np.arange(len(CITIES)) + i * width,
                [aucs[city][str(use_cuda)][model_name]["time"] for city in CITIES],
                width=width,
                color=PLOT_COLORS[model_name] if model_name in PLOT_COLORS else "grey",
                label=model_name,
            )
            # Write the values of the bars
            for j, city in enumerate(CITIES):
                plt.text(
                    j + i * width - 0.05,
                    aucs[city][str(use_cuda)][model_name]["time"],
                    f"{aucs[city][str(use_cuda)][model_name]['time']:.2f}",
                    fontdict={"size": 8},
                )
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + f"inference_times_cuda_{use_cuda}.png")

        # Same for the emissions
        plt.figure(figsize=(16, 7))
        plt.rcParams.update({"font.size": 17})
        plt.title(f"Inference emissions {'(CUDA)' if use_cuda else '(CPU)'}")
        plt.xlabel("City")
        plt.ylabel("Emissions (gCO2e)")
        plt.xticks(np.arange(len(CITIES)), CITIES)
        for i, model_name in enumerate(MODELS):
            plt.bar(
                np.arange(len(CITIES)) + i * width,
                [aucs[city][str(use_cuda)][model_name]["emissions"] for city in CITIES],
                width=width,
                color=PLOT_COLORS[model_name] if model_name in PLOT_COLORS else "grey",
                label=model_name,
            )
            # Write the values of the bars with one significant digit
            for j, city in enumerate(CITIES):
                plt.text(
                    j + i * width - 0.05,
                    aucs[city][str(use_cuda)][model_name]["emissions"],
                    f"{aucs[city][str(use_cuda)][model_name]['emissions']:.3f}",
                    fontdict={"size": 8},
                )
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + f"inference_emissions_cuda_{use_cuda}.png")
