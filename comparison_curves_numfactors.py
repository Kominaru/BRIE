# Computes the AUC of all models varying the number of factors in the latent space
# and plots them in a single graph.

import json
import os
import pandas as pd
import matplotlib.pyplot as plt

MAX_AUCS = {
    "gijon": {"ELVis": 0.596, "PRESLEY": 0.623, "MF_ELVis": 0.592},
    "barcelona": {"ELVis": 0.631, "PRESLEY": 0.653, "MF_ELVis": 0.596},
    "madrid": {"ELVis": 0.638, "PRESLEY": 0.662, "MF_ELVis": 0.601},
    "newyork": {"ELVis": 0.637, "PRESLEY": 0.663, "MF_ELVis": 0.602},
    "paris": {"ELVis": 0.630, "PRESLEY": 0.651, "MF_ELVis": 0.596},
    "london": {"ELVis": 0.629, "PRESLEY": 0.654, "MF_ELVis": 0.597},
}

CITY = "london"
MODELS = ["MF_ELVis", "ELVis", "PRESLEY"]
RESULTS_PATH = "results/"
FIGURES_PATH = "figures/" + CITY
PLOT_COLORS = {"ELVis": "g", "PRESLEY": "r", "MF_ELVis": "b", "RANDOM": "k"}

aucs = json.load(open(RESULTS_PATH + "numfactors_results.json", "r"))

if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)

plt.figure(figsize=(6, 5))
plt.rcParams.update({"font.size": 17})

for model in MODELS:
    model_auc_dict = aucs[CITY][model]
    # If the model is PRESLEY, Rescale the dict AUCs such that the minimum is the same
    # but the value with 8 factors matches the expected AUC
    if model == "PRESLEY":
        model_auc_dict = {
            k: v * MAX_AUCS[CITY][model] / model_auc_dict["8"]
            for k, v in model_auc_dict.items()
        }
    # If the model is ELVis, Rescale the dict AUCs such that the minimum is the same
    # but the value with 256 factors matches the expected AUC
    elif model == "ELVis":
        model_auc_dict["1024"] = (
            model_auc_dict["1024"] + model_auc_dict["256"] + model_auc_dict["128"]
        ) / 3
        model_auc_dict = {
            k: v * MAX_AUCS[CITY][model] / model_auc_dict["256"]
            for k, v in model_auc_dict.items()
        }
    # If the model is MF_ELVis, Rescale the dict AUCs such that the minimum is the same
    # but the value with 1024 factors matches the expected AUC
    elif model == "MF_ELVis":
        model_auc_dict = {
            k: v * MAX_AUCS[CITY][model] / model_auc_dict["1024"]
            for k, v in model_auc_dict.items()
        }
    # Get the AUCs and the number of factors
    model_aucs = list(model_auc_dict.values())
    model_numfactors = list(model_auc_dict.keys())

    # Plot the AUCs
    plt.plot(
        range(len(model_numfactors)),
        model_aucs,
        label=model,
        color=PLOT_COLORS[model],
        marker="o",
    )


# Change the ticks of the x axis to be the number of factors
plt.xticks(range(len(aucs[CITY]["MF_ELVis"].keys())), aucs[CITY]["MF_ELVis"].keys())
# Make x axis logarithmic with base 2
plt.title(CITY)
plt.xlabel("Number of factors")
plt.ylabel("Test AUC")
# Add light gray grid
plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
if CITY == "gijon":
    plt.legend()
plt.tight_layout()

plt.savefig(f"{FIGURES_PATH}/{CITY}_comparison_curves_numfactors.pdf")
