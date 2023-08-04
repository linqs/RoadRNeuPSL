import logging
import os
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logger
import utils

from torch.utils.data import DataLoader

from data.RoadRDataset import RoadRDataset
from models.roadr_detr_neupsl import RoadRDETRNeuPSL
from models.trainer import Trainer
from utils import BASE_RESULTS_DIR
from utils import make_dir

LOGGING_LEVEL = logging.DEBUG
SEED = 42

TASK_NAME = "task1"
LABELED_VIDEOS = ["2014-07-14-14-49-50_stereo_centre_01",
                  "2015-02-03-19-43-11_stereo_centre_04",
                  "2015-02-24-12-32-19_stereo_centre_04"]

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_FILE_NAME = 'road_trainval_v1.0.json'
DATA_DIR = os.path.join(THIS_DIR, "../data")

MAX_FRAMES = 1000


def main():
    utils.seed_everything(42)

    logger.initLogging(LOGGING_LEVEL)

    make_dir(BASE_RESULTS_DIR)
    make_dir(os.path.join(BASE_RESULTS_DIR, TASK_NAME))

    # Load training dataset.
    logging.info("Loading training dataset.")
    dataset = RoadRDataset(LABELED_VIDEOS, os.path.join(DATA_DIR, DATA_FILE_NAME), max_frames=MAX_FRAMES)
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # TODO(Charles): Create a separate validation dataset.
    validation_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Preprocess training dataset.

    # Build Model.
    # TODO(connor): Change to shell call to PSL.
    model = RoadRDETRNeuPSL()
    model.internal_init_model(None)

    # Train model.
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    trainer = Trainer(model.model, optimizer, lr_scheduler, utils.get_torch_device(), os.path.join(BASE_RESULTS_DIR, TASK_NAME))
    trainer.train(train_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
