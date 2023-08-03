import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logger
import utils

from data.RoadRDataset import RoadRDataset
from models.roadr_detr_neupsl import RoadRDETRNeuPSL

LOGGING_LEVEL = logging.DEBUG
SEED = 42

TASK_NAME = "task1"
LABELED_VIDEOS = ["2014-07-14-14-49-50_stereo_centre_01",
                  "2015-02-03-19-43-11_stereo_centre_04",
                  "2015-02-24-12-32-19_stereo_centre_04"]

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

DATA_FILE_NAME = 'road_trainval_v1.0.json'
DATA_DIR = os.path.join(THIS_DIR, "../data")

MAX_FRAMES = 100


def main():
    utils.seed_everything(42)

    logger.initLogging(LOGGING_LEVEL)

    # Load training dataset.
    logging.info("Loading training dataset.")
    train_dataset = RoadRDataset(LABELED_VIDEOS, os.path.join(DATA_DIR, DATA_FILE_NAME), max_frames=MAX_FRAMES)

    # Preprocess training dataset.

    # Build Model.
    # TODO(connor): Change to shell call to PSL.
    model = RoadRDETRNeuPSL()
    model.internal_init_model(None)

    # Make a prediction.
    frames, train_images, labels, boxes = train_dataset[:2]

    predictions = model.internal_predict(train_images, {"learn": True})

    # print("predictions: ", predictions)

    # Evaluate the model.
    model.internal_eval(train_dataset[:2], {"learn": True})


if __name__ == "__main__":
    main()
