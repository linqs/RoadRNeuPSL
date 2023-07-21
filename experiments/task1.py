import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils

from data.RoadRDataset import RoadRDataset

SEED = 42

TASK_NAME = "task1"
LABELED_VIDEOS = ["2014-07-14-14-49-50_stereo_centre_01",
                  "2015-02-03-19-43-11_stereo_centre_04",
                  "2015-02-24-12-32-19_stereo_centre_04"]

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

DATA_FILE_NAME = 'road_trainval_v1.0.json'
DATA_DIR = os.path.join(THIS_DIR, "../data")


def main():
    utils.seed_everything(42)

    # Load training dataset.
    train_dataset = RoadRDataset(LABELED_VIDEOS, os.path.join(DATA_DIR, DATA_FILE_NAME))

    # Preprocess training dataset.

    # Build Model.
    # model =


if __name__ == "__main__":
    main()
