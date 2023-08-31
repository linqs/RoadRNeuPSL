import csv

import numpy as np
import os
import random
import torch

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_RESULTS_DIR = os.path.join(THIS_DIR, "results")

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.csv"
TRAINED_MODEL_FILENAME = "trained_model_parameters.pt"
TRAINING_CONVERGENCE_FILENAME = "training_convergence.csv"
TRAINING_SUMMARY_FILENAME = "training_summary.csv"
TEST_EVALUATION_FILENAME = "test_evaluation.csv"


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


def write_psl_file(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")


def load_csv_file(path, delimiter=','):
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        return list(reader)


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
