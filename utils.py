import csv
import json
import pickle

import numpy as np
import os
import random
import torch

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_RESULTS_DIR = os.path.join(THIS_DIR, "results")
BASE_DATA_DIR = os.path.join(THIS_DIR, "data")

TRAIN_VALIDATION_DATA_PATH = os.path.join(BASE_DATA_DIR, "road_trainval_v1.0.json")

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.csv"
EVALUATION_SUMMARY_FILENAME = "evaluation_summary.json"
EVALUATION_PREDICTION_JSON_FILENAME = "evaluation_predictions.json"
EVALUATION_PREDICTION_PKL_FILENAME = "evaluation_predictions.pkl"
EVALUATION_METRICS_FILENAME = "evaluation_metrics.csv"
TRAINED_MODEL_DIR = "trained_model"
TRAINED_MODEL_FILENAME = "trained_model_parameters.pt"
TRAINING_CONVERGENCE_FILENAME = "training_convergence.csv"
TRAINING_SUMMARY_FILENAME = "training_summary.csv"


def check_cached_file(out_file: str):
    """
    Check if the file with the provided path already exists.
    :param out_file: The path to the file.
    :return: True if the output directory contains an out.txt file indicating the experiment has been ran.
    """
    return os.path.exists(out_file)


def make_dir(out_directory: str):
    """
    Make the run output directory. If the directory exists, do nothing.
    :param out_directory: The path to the run output directory.
    """
    os.makedirs(out_directory, exist_ok=True)


def load_csv_file(path, delimiter=','):
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        return list(reader)


def write_json_file(path, data, indent=0):
    with open(path, "w") as file:
        if indent is None:
            json.dump(data, file)
        else:
            json.dump(data, file, indent=indent)


def load_json_file(path):
    with open(path, "r") as file:
        return json.load(file)


def write_pkl_file(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def write_psl_file(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")


def load_psl_file(path, dtype=str):
    data = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '':
                continue

            data.append(list(map(dtype, line.split("\t"))))

    return data


def enumerate_hyperparameters(hyperparameters_dict, current_hyperparameters={}):
    for key in sorted(hyperparameters_dict):
        hyperparameters = []
        for value in hyperparameters_dict[key]:
            next_hyperparameters = current_hyperparameters.copy()
            next_hyperparameters[key] = value

            remaining_hyperparameters = hyperparameters_dict.copy()
            remaining_hyperparameters.pop(key)

            if remaining_hyperparameters:
                hyperparameters = hyperparameters + enumerate_hyperparameters(remaining_hyperparameters, next_hyperparameters)
            else:
                hyperparameters.append(next_hyperparameters)
        return hyperparameters


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
