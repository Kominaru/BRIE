# Compares the validation AUC curves for the different models of a tune run of a city
#
# Usage:
# python compare_tune_curves.py --city barcelona --model PRESLEY --lr 5e-4 --max_epochs 50 --dropout 0.2 -d 8

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--city", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--lr", type=float)
parser.add_argument("--max_epochs", type=int)
parser.add_argument("--dropout", type=float)
parser.add_argument("--max_d", type=int)
args = parser.parse_args()

CITY = args.city
MODEL = args.model
LR = args.lr
MAX_EPOCHS = args.max_epochs
DROPOUT = args.dropout

EXPECTED_AUC = {
    "gijon": {"ELVis": 0.596, "PRESLEY": 0.625, "MF_ELVis": 0.592},
    "barcelona": {"ELVis": 0.631, "PRESLEY": 0.650, "MF_ELVis": 0.596},
    "madrid": {"ELVis": 0.638, "PRESLEY": 0.661, "MF_ELVis": 0.601},
    "newyork": {"ELVis": 0.637, "PRESLEY": 0.663, "MF_ELVis": 0.602},
    "paris": {"ELVis": 0.630, "PRESLEY": 0.649, "MF_ELVis": 0.596},
    "london": {"ELVis": 0.629, "PRESLEY": 0.647, "MF_ELVis": 0.597},
}


# Path: compare_tune_curves.py

trained_models = []

# From all the folders in the tune directory for the city and model, get the hyperparameters and the validation AUC values
for folder in os.listdir("csv_logs/" + CITY + "_tune" + "/" + MODEL):
    # Read hyperparams from yaml file as a dict
    with open(
        "csv_logs/" + CITY + "_tune" + "/" + MODEL + "/" + folder + "/hparams.yaml"
    ) as f:
        hyperparams = yaml.load(f, Loader=yaml.FullLoader)

    # If metrics.csv does not exist, skip this folder
    if not os.path.exists(
        "csv_logs/" + CITY + "_tune" + "/" + MODEL + "/" + folder + "/metrics.csv"
    ):
        continue
    # Read validation AUC values from csv file as a list
    df = pd.read_csv(
        "csv_logs/" + CITY + "_tune" + "/" + MODEL + "/" + folder + "/metrics.csv"
    )

    # Get the validation AUC values (only the odd rows because the even rows are train metrics and have no AUC)
    val_auc = df["val_auc"].iloc[::2].tolist()

    # Add the hyperparams and validation AUC values to the list of trained models
    trained_models.append(
        {
            "hyperparams": hyperparams,
            "val_auc": val_auc,
        }
    )

    # Print the hyperparams and the maximum validation AUC of this model configuration
    print(
        hyperparams,
        max(val_auc),
    )

# From all the trained models, keep only those with the same hyperparameters as the ones specified in the arguments
# (except for the city and model, and the maximum number of factors)
trained_models = [
    model
    for model in trained_models
    if all(
        [
            model["hyperparams"][key] == value
            for key, value in vars(args).items()
            if key not in ["city", "model", "max_d", "max_epochs"] and value is not None
        ]
    )
]

# Keep only the models that have equal or less factors than the maximum number of factors specified in the arguments
trained_models = [
    model
    for model in trained_models
    if model["hyperparams"]["d"] <= (args.max_d or model["hyperparams"]["d"])
]

# Keep only the models that have the maximum validation AUC of all models in at least one epoch

# Get the maximum validation AUC of all models in each epoch
max_val_auc = [
    max(
        [
            model["val_auc"][epoch]
            for model in trained_models
            if len(model["val_auc"]) > epoch
        ]
    )
    for epoch in range(max([len(model["val_auc"]) for model in trained_models] + [0]))
]

# Get the models that have the maximum validation AUC of all models in at least one epoch
trained_models = [
    model
    for model in trained_models
    if max(model["val_auc"]) > 0.658
    # if any(
    #     [model["val_auc"][i] == max_val_auc[i] for i in range(len(model["val_auc"]))]
    # )
    # or
]

# X values are the epochs multiplied by 4/7
# x = [i * 4 / 7 for i in range(len(max_val_auc))]
x = [i for i in range(len(max_val_auc))]

# Plot the validation AUC curves of the models
for model in trained_models:
    plt.plot(x[: len(model["val_auc"])], model["val_auc"], label=model["hyperparams"])

# Plot the expected validation AUC for the city in ELVis and MF_ELVis
plt.plot(
    [EXPECTED_AUC[CITY]["ELVis"]] * 100,
    label="ELVis AUC",
    linestyle="dashed",
)

plt.plot(
    [EXPECTED_AUC[CITY]["MF_ELVis"]] * 100,
    label="MF_ELVis AUC",
    linestyle="dashed",
)

# See x axis only from 0 to the maximum number of epochs
plt.xlim(0, len(max_val_auc) * 4 / 7)

# Make x axis ticks every 10 epochs
# plt.xticks([i for i in range(0, int(len(max_val_auc) * (4 / 7) + 1), 10)])

plt.xlabel("Epoch")
plt.ylabel("Validation AUC")
plt.legend()
plt.show()
