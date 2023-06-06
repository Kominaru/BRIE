# Script to call through command line to train models

import os

# Path: train_models.py


def train_model_through_command_line(model_name: str, city: str, d: int):
    command = (
        "python main.py --stage train --model "
        + model_name
        + " --city "
        + city
        + " --use_train_val "
        + " -d "
        + str(d)
        + " --no_validation"
    )

    if model_name == "ELVis":
        command += " --lr 5e-4 --batch_size 32768 --max_epochs 100"

    elif model_name == "PRESLEY":
        command += " --lr 1e-3 --batch_size 16384 --max_epochs 25 --dropout 0.5"

    elif model_name == "MF_ELVis":
        command += " --lr 5e-4 --batch_size 32768 --max_epochs 25"

    if city in ["barcelona", "gijon", "madrid"]:
        command += " --workers 4"
    elif city in ["paris", "newyork"]:
        command += " --workers 2"
    elif city in ["london"]:
        command += " --workers 1"

    print(command)
    os.system(command)

    test_command = (
        "python main.py --stage test --model "
        + model_name
        + " --city "
        + city
        + " --use_train_val "
        + " -d "
        + str(d)
        + " --no_validation"
    )

    if city in ["barcelona", "gijon", "madrid"]:
        test_command += " --workers 4"
    elif city in ["paris", "newyork"]:
        test_command += " --workers 2"
    elif city in ["london"]:
        test_command += " --workers 1"

    os.system(test_command)


if __name__ == "__main__":
    # Cities to train
    cities = ["gijon"]
    # Models to train
    models = ["MF_ELVis", "ELVis", "PRESLEY"]
    for city in cities:
        for model in models:
            for d in [4, 8, 32, 128, 256, 1024]:
                train_model_through_command_line(model, city, d)
