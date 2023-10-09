import csv
import json
import pickle

import numpy as np
import os
import random
import torch

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_CLI_DIR = os.path.join(THIS_DIR, "cli")
BASE_DATA_DIR = os.path.join(THIS_DIR, "data")
BASE_RESULTS_DIR = os.path.join(THIS_DIR, "results")

PSL_MODELS_DIR = os.path.join(THIS_DIR, "models", "psl")

BASE_RGB_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "rgb-images")
BASE_TEST_RGB_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "test-rgb-images")
BASE_RGB_IMAGES_PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "rgb-images-processed")

TRAIN_VALIDATION_DATA_PATH = os.path.join(BASE_DATA_DIR, "road_trainval_v1.0.json")
CONSTRAINTS_PATH = os.path.join(BASE_DATA_DIR, "constraints", "hard-co-occurrence.csv")

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.csv"
EVALUATION_SUMMARY_FILENAME = "evaluation_summary.json"
PREDICTION_LOGITS_JSON_FILENAME = "prediction_logits.json"
PREDICTION_LOGITS_PKL_FILENAME = "prediction_logits.pkl"
PREDICTION_LABELS_JSON_FILENAME = "prediction_labels.json"
PREDICTION_LABELS_PKL_FILENAME = "prediction_labels.pkl"
EVALUATION_METRICS_FILENAME = "evaluation_metrics.json"
NEURAL_TRAINED_MODEL_DIR = "trained_model"
NEURAL_TRAINED_MODEL_FILENAME = "trained_model_parameters.pt"
NEURAL_TRAINING_CONVERGENCE_FILENAME = "training_convergence.csv"
NEURAL_VALIDATION_CONVERGENCE_FILENAME = "validation_convergence.csv"
NEURAL_TRAINING_SUMMARY_FILENAME = "training_summary.csv"
NEURAL_VALIDATION_SUMMARY_FILENAME = "validation_summary.csv"
NEURAL_VALID_INFERENCE_DIR = "neural_valid_inference"
NEURAL_TEST_INFERENCE_DIR = "neural_test_inference"
NEUPSL_MODEL_FILENAME = "roadr.json"
NEUPSL_TRAINED_MODEL_DIR = "neupsl_trained_model"
NEUPSL_TRAINED_MODEL_FILENAME = "neupsl_trained_model_parameters.pt"
NEUPSL_TRAINING_CONVERGENCE_FILENAME = "neupsl_training_convergence.csv"
NEUPSL_TRAINING_SUMMARY_FILENAME = "neupsl_training_summary.csv"
NEUPSL_VALID_INFERENCE_DIR = "neupsl_valid_inference"
NEUPSL_TEST_INFERENCE_DIR = "neupsl_test_inference"

SEED = 4

IMAGE_HEIGHT = 960
IMAGE_WIDTH = 1280
NUM_CLASSES = 41
NUM_QUERIES = 100
NUM_NEUPSL_QUERIES = 20

BOX_CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5
LABEL_CONFIDENCE_THRESHOLD = 0.025
NUM_SAVED_IMAGES = 10

VIDEO_PARTITIONS = {
    "task1": {
        "TRAIN": ["2014-07-14-14-49-50_stereo_centre_01",
                  "2015-02-03-19-43-11_stereo_centre_04",
                  "2015-02-24-12-32-19_stereo_centre_04"],
        "VALID": ["2014-06-26-09-53-12_stereo_centre_02",
                  "2014-11-25-09-18-32_stereo_centre_04",
                  "2015-02-13-09-16-26_stereo_centre_02"],
        "TEST": ["2014-06-26-09-31-18_stereo_centre_02",
                 "2014-12-10-18-10-50_stereo_centre_02",
                 "2015-02-03-08-45-10_stereo_centre_04",
                 "2015-02-06-13-57-16_stereo_centre_01"]
    },
    "task2": {
        "TRAIN": ["2014-06-25-16-45-34_stereo_centre_02",
                  "2014-07-14-14-49-50_stereo_centre_01",
                  "2014-07-14-15-42-55_stereo_centre_03",
                  "2014-08-08-13-15-11_stereo_centre_01",
                  "2014-08-11-10-59-18_stereo_centre_02",
                  "2014-11-14-16-34-33_stereo_centre_06",
                  "2014-11-18-13-20-12_stereo_centre_05",
                  "2014-11-21-16-07-03_stereo_centre_01",
                  "2014-12-09-13-21-02_stereo_centre_01",
                  "2015-02-03-08-45-10_stereo_centre_02",
                  "2015-02-03-19-43-11_stereo_centre_04",
                  "2015-02-06-13-57-16_stereo_centre_02",
                  "2015-02-13-09-16-26_stereo_centre_05",
                  "2015-02-24-12-32-19_stereo_centre_04",
                  "2015-03-03-11-31-36_stereo_centre_01"],
        "VALID": ["2014-06-26-09-53-12_stereo_centre_02",
                  "2014-11-25-09-18-32_stereo_centre_04",
                  "2015-02-13-09-16-26_stereo_centre_02"],
        "TEST": ["2014-06-26-09-31-18_stereo_centre_02",
                 "2014-12-10-18-10-50_stereo_centre_02",
                 "2015-02-03-08-45-10_stereo_centre_04",
                 "2015-02-06-13-57-16_stereo_centre_01"]
    }
}

LABEL_TYPES = ['agent', 'action', 'loc']

LABEL_MAPPING = {
    0: "Ped",
    1: "Car",
    2: "Cyc",
    3: "Mobike",
    4: "MedVeh",
    5: "LarVeh",
    6: "Bus",
    7: "EmVeh",
    8: "TL",
    9: "OthTL",
    10: "Red",
    11: "Amber",
    12: "Green",
    13: "MovAway",
    14: "MovTow",
    15: "Mov",
    16: "Brake",
    17: "Stop",
    18: "IncatLft",
    19: "IncatRgt",
    20: "HazLit",
    21: "TurLft",
    22: "TurRht",
    23: "Ovtak",
    24: "Wait2X",
    25: "XingFmLft",
    26: "XingFmRht",
    27: "Xing",
    28: "PushObj",
    29: "VehLane",
    30: "OutgoLane",
    31: "OutgoCycLane",
    32: "IncomLane",
    33: "IncomCycLane",
    34: "Pav",
    35: "LftPav",
    36: "RhtPav",
    37: "Jun",
    38: "xing",
    39: "BusStop",
    40: "parking"
}

ORIGINAL_LABEL_MAPPING = {
    "agent": {
        0: [0, "Ped"],
        1: [1, "Car"],
        2: [2, "Cyc"],
        3: [3, "Mobike"],
        5: [4, "MedVeh"],
        6: [5, "LarVeh"],
        7: [6, "Bus"],
        8: [7, "EmVeh"],
        9: [8, "TL"],
        10: [9, "OthTL"]
    },
    "action": {
        0: [10, "Red"],
        1: [11, "Amber"],
        2: [12, "Green"],
        3: [13, "MovAway"],
        4: [14, "MovTow"],
        5: [15, "Mov"],
        7: [16, "Brake"],
        8: [17, "Stop"],
        9: [18, "IncatLft"],
        10: [19, "IncatRht"],
        11: [20, "HazLit"],
        12: [21, "TurLft"],
        13: [22, "TurRht"],
        16: [23, "Ovtak"],
        17: [24, "Wait2X"],
        18: [25, "XingFmLft"],
        19: [26, "XingFmRht"],
        20: [27, "Xing"],
        21: [28, "PushObj"]
    },
    "loc": {
        0: [29, "VehLane"],
        1: [30, "OutgoLane"],
        2: [31, "OutgoCycLane"],
        3: [32, "IncomLane"],
        4: [33, "IncomCycLane"],
        5: [34, "Pav"],
        6: [35, "LftPav"],
        7: [36, "RhtPav"],
        8: [37, "Jun"],
        9: [38, "xing"],
        10: [39, "BusStop"],
        11: [40, "parking"]
    }
}


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
