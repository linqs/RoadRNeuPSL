import argparse
import logging
import os
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logger
import utils

from torch.utils.data import DataLoader

from data.roadr_dataset import RoadRDataset
from experiments.task_models import build_task_1_model
from models.trainer import Trainer
from utils import BASE_RESULTS_DIR

torch.cuda.empty_cache()

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(THIS_DIR, "..", "data", "road_trainval_v1.0.json")

TASK_NAME = "task1"

LABELED_VIDEOS = ["2014-07-14-14-49-50_stereo_centre_01",
                  "2015-02-03-19-43-11_stereo_centre_04",
                  "2015-02-24-12-32-19_stereo_centre_04"]

HYPERPARAMETERS = {
    'learning-rate': [1.0e-3, 1.0e-4, 1.0e-5],
    'weight-decay': [1.0e-4, 1.0e-5],
    'batch-size': [6],
    'dropout': [0.1, 0.2],
    'step-size': [200, 400],
    'gamma': [0.1, 0.2],
    'epochs': [100]
}

DEFAULT_PARAMETERS = {
    'learning-rate': 1.0e-4,
    'weight-decay': 1.0e-4,
    'batch-size': 6,
    'dropout': 0.1,
    'step-size': 200,
    'gamma': 0.1,
    'epochs': 100
}


def run_setting(dataset, parameters, parameters_string):
    os.makedirs(os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string), exist_ok=True)

    train_dataloader = DataLoader(dataset, batch_size=parameters['batch-size'], shuffle=True)
    validation_dataloader = DataLoader(dataset, batch_size=parameters['batch-size'], shuffle=True)

    model = build_task_1_model(dropout=parameters["dropout"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning-rate"], weight_decay=parameters["weight-decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameters["step-size"], gamma=parameters["gamma"])

    trainer = Trainer(model, optimizer, lr_scheduler, utils.get_torch_device(), os.path.join(BASE_RESULTS_DIR, TASK_NAME, parameters_string))
    trainer.train(train_dataloader, validation_dataloader, n_epochs=parameters["epochs"])

    return trainer.compute_validation_score(validation_dataloader)


def main(arguments):
    utils.seed_everything(arguments.seed)

    logger.initLogging(arguments.log_level)
    logging.info("Beginning task 1 experiment.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    hyperparameters = utils.enumerate_hyperparameters(HYPERPARAMETERS)

    best_loss = float('inf')
    best_parameter_string = ''
    parameter_setting = DEFAULT_PARAMETERS

    dataset = RoadRDataset(LABELED_VIDEOS, DATA_PATH, max_frames=arguments.max_frames)

    if arguments.hyperparameter_search:
        for index in range(len(hyperparameters)):
            hyperparameters_string = ''
            for key in sorted(hyperparameters[index].keys()):
                hyperparameters_string = hyperparameters_string + key + ':' + str(hyperparameters[index][key]) + ' -- '
            logging.info("\n%d \ %d -- %s" % (index, len(hyperparameters), hyperparameters_string[:-3]))

            loss = run_setting(dataset, hyperparameters[index], hyperparameters_string)

            if loss < best_loss:
                best_loss = loss
                best_parameter_string = hyperparameters_string
                parameter_setting = hyperparameters[index]

            logging.info("Best hyperparameter setting: %s with loss %f" % (best_parameter_string, best_loss))

    run_setting(dataset, parameter_setting, 'final')


def _load_args():
    parser = argparse.ArgumentParser(description='Generate Road-R PSL data.')

    parser.add_argument('--seed', dest='seed',
                        action='store', type=int, default=4,
                        help='Seed for random number generator.')
    parser.add_argument('--max-frames', dest='max_frames',
                        action='store', type=int, default=1000,
                        help='Maximum number of frames to use from all videos.')
    parser.add_argument('--hyperparameter-search', dest='hyperparameter_search',
                        action='store', type=bool, default=False,
                        help='Run hyperparameter search.')
    parser.add_argument('--log-level', dest='log_level',
                        action='store', type=str, default='INFO',
                        help='Logging level.', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
