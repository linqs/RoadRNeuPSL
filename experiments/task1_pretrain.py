import argparse
import logging
import os
import sys

import torch

from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger
import utils

from data.stream_roadr_dataset import RoadRDataset
from models.trainer import Trainer
from utils import BASE_RESULTS_DIR
from utils import NUM_CLASSES
from utils import SEED
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import TRAINED_MODEL_DIR
from utils import TRAINED_MODEL_FILENAME
from utils import TRAINING_SUMMARY_FILENAME
from utils import VIDEO_PARTITIONS

TASK_NAME = "task1"

TRAIN_VIDEOS = VIDEO_PARTITIONS[TASK_NAME]["TRAIN"]

HYPERPARAMETERS = {
    "learning-rate": [1.0e-4, 1.0e-5],
    "weight-decay": [1.0e-5],
    "batch-size": [2],
    "step-size": [100],
    "gamma": [1.0],
    "epochs": [100]
}

DEFAULT_PARAMETERS = {
    "learning-rate": 1.0e-6,
    "weight-decay": 1.0e-5,
    "batch-size": 2,
    "step-size": 500,
    "gamma": 1.0,
    "epochs": 10
}


def task_1_model():
    return DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50-dc5",
                                                  num_labels=NUM_CLASSES,
                                                  ignore_mismatched_sizes=True).to(utils.get_torch_device())


def run_setting(arguments, train_dataset, parameters, parameters_string):
    if os.path.isfile(os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string, TRAINING_SUMMARY_FILENAME)) and not arguments.resume_from_checkpoint:
        logging.info("Skipping training for %s, already exists." % (parameters_string,))
        return float(utils.load_csv_file(os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string, TRAINING_SUMMARY_FILENAME))[1][0])

    os.makedirs(os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string), exist_ok=True)

    train_dataloader = DataLoader(train_dataset, batch_size=parameters["batch-size"], shuffle=True, num_workers=int(os.cpu_count()) - 2, prefetch_factor=4, persistent_workers=True)

    model = task_1_model()

    if arguments.resume_from_checkpoint:
        logging.info("Loading model from checkpoint: %s" % (os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string, TRAINED_MODEL_FILENAME),))
        model.load_state_dict(torch.load(os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string, TRAINED_MODEL_FILENAME)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning-rate"], weight_decay=parameters["weight-decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameters["step-size"], gamma=parameters["gamma"])

    trainer = Trainer(model, optimizer, lr_scheduler, utils.get_torch_device(), os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string))
    trainer.train(train_dataloader, n_epochs=parameters["epochs"])

    return float(utils.load_csv_file(os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string, TRAINING_SUMMARY_FILENAME))[1][0])


def main(arguments):
    utils.seed_everything(arguments.seed)

    logger.initLogging(arguments.log_level)
    logging.info("Beginning pre-training task 1.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    hyperparameters = utils.enumerate_hyperparameters(HYPERPARAMETERS)

    best_loss = float("inf")
    best_parameter_string = ""
    parameter_setting = DEFAULT_PARAMETERS

    if arguments.hyperparameter_search:
        logging.info("Loading Training Dataset")
        train_dataset = RoadRDataset(TRAIN_VIDEOS, TRAIN_VALIDATION_DATA_PATH, arguments.image_resize, max_frames=arguments.max_frames)

        for index in range(len(hyperparameters)):
            hyperparameters_string = ""
            for key in sorted(hyperparameters[index].keys()):
                hyperparameters_string = hyperparameters_string + key + ":" + str(hyperparameters[index][key]) + " -- "
            logging.info("\n%d \ %d -- %s" % (index, len(hyperparameters), hyperparameters_string[:-4]))

            loss = run_setting(arguments, train_dataset, hyperparameters[index], hyperparameters_string[:-4])

            if loss < best_loss:
                best_loss = loss
                best_parameter_string = hyperparameters_string[:-4]
                parameter_setting = hyperparameters[index]

            logging.info("Best hyperparameter setting: %s with loss %f" % (best_parameter_string, best_loss))

    logging.info("Loading Training Dataset")
    train_dataset = RoadRDataset(TRAIN_VIDEOS, TRAIN_VALIDATION_DATA_PATH, arguments.image_resize, max_frames=arguments.max_frames)

    loss = run_setting(arguments, train_dataset, parameter_setting, TRAINED_MODEL_DIR)
    logging.info("Final loss: %f" % (loss,))


def _load_args():
    parser = argparse.ArgumentParser(description="RoadR Task 1 Pre-Training Network")

    parser.add_argument("--seed", dest="seed",
                        action="store", type=int, default=SEED,
                        help="Seed for random number generator.")
    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default="INFO",
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--image-resize", dest="image_resize",
                        action="store", type=float, default=1.0,
                        help="Resize factor for all images.")
    parser.add_argument("--max-frames", dest="max_frames",
                        action="store", type=int, default=0,
                        help="Maximum number of frames to use from each videos. Default is 0, which uses all frames.")
    parser.add_argument("--hyperparameter-search", dest="hyperparameter_search",
                        action="store", type=bool, default=False,
                        help="Run hyperparameter search.")
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint",
                        action="store", type=bool, default=False,
                        help="Resume training from the most recent checkpoint.")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
