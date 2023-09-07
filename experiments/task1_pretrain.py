import argparse
import logging
import os
import sys

import torch
import torchvision

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger
import utils

from data.roadr_dataset import RoadRDataset
from models.detr import DETR
from models.trainer import Trainer
from utils import BASE_RESULTS_DIR
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import TRAINED_MODEL_DIR
from utils import TRAINING_SUMMARY_FILENAME


TASK_NAME = "task1"

TRAIN_VIDEOS = ["2014-07-14-14-49-50_stereo_centre_01",
                "2015-02-03-19-43-11_stereo_centre_04",
                "2015-02-24-12-32-19_stereo_centre_04"]

HYPERPARAMETERS = {
    "learning-rate": [1.0e-3, 1.0e-4, 1.0e-5],
    "weight-decay": [1.0e-4, 1.0e-5],
    "batch-size": [32],
    "dropout": [0.1, 0.2],
    "step-size": [200, 400],
    "gamma": [0.1, 0.2],
    "epochs": [100]
}

DEFAULT_PARAMETERS = {
    "learning-rate": 1.0e-4,
    "weight-decay": 1.0e-5,
    "batch-size": 6,
    "dropout": 0.1,
    "step-size": 400,
    "gamma": 0.2,
    "epochs": 100
}


def task_1_model(dropout, image_resize, num_queries):
    resnet34 = torchvision.models.resnet34(weights=None)

    # Remove the last two layers of the resnet34 model.
    # The last layer is a fully connected layer and the second to last layer is a pooling layer.
    backbone = torch.nn.Sequential(*list(resnet34.children())[:-2])

    transformer = torch.nn.Transformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=512,
        dropout=dropout,
        activation="relu",
        batch_first=True,
        norm_first=False
    )
    return DETR(backbone, transformer, image_resize=image_resize, num_queries=num_queries).to(utils.get_torch_device())


def run_setting(arguments, train_dataset, valid_dataset, parameters, parameters_string):
    if os.path.isfile(os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string, TRAINING_SUMMARY_FILENAME)):
        logging.info("Skipping training for %s, already exists." % (parameters_string,))
        return float(utils.load_csv_file(os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string, TRAINING_SUMMARY_FILENAME))[1][1])

    os.makedirs(os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string), exist_ok=True)

    train_dataloader = DataLoader(train_dataset, batch_size=parameters["batch-size"], shuffle=True)
    validation_dataloader = DataLoader(valid_dataset, batch_size=parameters["batch-size"], shuffle=True)

    model = task_1_model(parameters["dropout"], arguments.image_resize, arguments.num_queries)

    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning-rate"], weight_decay=parameters["weight-decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameters["step-size"], gamma=parameters["gamma"])

    trainer = Trainer(model, optimizer, lr_scheduler, utils.get_torch_device(), os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string))
    trainer.train(train_dataloader, validation_dataloader, n_epochs=parameters["epochs"])

    return trainer.compute_validation_score(validation_dataloader)


def main(arguments):
    utils.seed_everything(arguments.seed)

    logger.initLogging(arguments.log_level)
    logging.info("Beginning pre-training task 1.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    hyperparameters = utils.enumerate_hyperparameters(HYPERPARAMETERS)

    best_loss = float("inf")
    best_parameter_string = ""
    parameter_setting = DEFAULT_PARAMETERS

    if arguments.hyperparameter_search:
        logging.info("Loading Training Dataset")
        train_dataset = RoadRDataset(TRAIN_VIDEOS, TRAIN_VALIDATION_DATA_PATH, arguments.image_resize, arguments.num_queries, start_frame_percentage=0.0, end_frame_percentage=0.8, max_frames=arguments.max_frames)
        logging.info("Loading Validation Dataset")
        valid_dataset = RoadRDataset(TRAIN_VIDEOS, TRAIN_VALIDATION_DATA_PATH, arguments.image_resize, arguments.num_queries, start_frame_percentage=0.8, end_frame_percentage=1.0, max_frames=arguments.max_frames)

        for index in range(len(hyperparameters)):
            hyperparameters_string = ""
            for key in sorted(hyperparameters[index].keys()):
                hyperparameters_string = hyperparameters_string + key + ":" + str(hyperparameters[index][key]) + " -- "
            logging.info("\n%d \ %d -- %s" % (index, len(hyperparameters), hyperparameters_string[:-4]))

            loss = run_setting(arguments, train_dataset, valid_dataset, hyperparameters[index], hyperparameters_string[:-4])

            if loss < best_loss:
                best_loss = loss
                best_parameter_string = hyperparameters_string[:-4]
                parameter_setting = hyperparameters[index]

            logging.info("Best hyperparameter setting: %s with loss %f" % (best_parameter_string, best_loss))

    logging.info("Loading Training Dataset")
    train_dataset = RoadRDataset(TRAIN_VIDEOS, TRAIN_VALIDATION_DATA_PATH, arguments.image_resize, arguments.num_queries, start_frame_percentage=0.0, end_frame_percentage=0.95, max_frames=arguments.max_frames)
    logging.info("Loading Validation Dataset")
    valid_dataset = RoadRDataset(TRAIN_VIDEOS, TRAIN_VALIDATION_DATA_PATH, arguments.image_resize, arguments.num_queries, start_frame_percentage=0.95, end_frame_percentage=1.0, max_frames=arguments.max_frames)

    run_setting(arguments, train_dataset, valid_dataset, parameter_setting, TRAINED_MODEL_DIR)


def _load_args():
    parser = argparse.ArgumentParser(description="RoadR Task 1 Pre-Training Network")

    parser.add_argument("--seed", dest="seed",
                        action="store", type=int, default=4,
                        help="Seed for random number generator.")
    parser.add_argument("--image-resize", dest="image_resize",
                        action="store", type=float, default=1.0,
                        help="Resize factor for all images.")
    parser.add_argument("--num-queries", dest="num_queries",
                        action="store", type=int, default=20,
                        help="Number of object queries, ie detection slot, in a frame.")
    parser.add_argument("--max-frames", dest="max_frames",
                        action="store", type=int, default=0,
                        help="Maximum number of frames to use from each videos. Default is 0, which uses all frames.")
    parser.add_argument("--hyperparameter-search", dest="hyperparameter_search",
                        action="store", type=bool, default=False,
                        help="Run hyperparameter search.")
    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default="INFO",
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
