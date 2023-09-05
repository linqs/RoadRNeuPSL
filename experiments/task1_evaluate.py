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
from experiments.task1_pretrain import task_1_model
from utils import BASE_RESULTS_DIR

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(THIS_DIR, "..", "data", "road_trainval_v1.0.json")
SAVED_MODEL_PATH = os.path.join(BASE_RESULTS_DIR, "task1", "final", "trained_model_parameters.pt")

VALID_VIDEOS = ["2014-06-26-09-53-12_stereo_centre_02",
                "2014-11-25-09-18-32_stereo_centre_04",
                "2015-02-13-09-16-26_stereo_centre_02"]


def main(arguments):
    utils.seed_everything(arguments.seed)

    logger.initLogging(arguments.log_level)
    logging.info("Beginning evaluating task 1.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    dataset = RoadRDataset(VALID_VIDEOS, DATA_PATH, arguments.image_resize, arguments.num_queries, max_frames=arguments.max_frames)
    dataloader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False)

    logging.info("Building and loading pre-trained model.")
    model = task_1_model(image_resize=arguments.image_resize)
    model.load_state_dict(torch.load(arguments.saved_model_path))

    logging.info("Evaluating model.")



def _load_args():
    parser = argparse.ArgumentParser(description='Evaluating Task 1.')

    parser.add_argument('--seed', dest='seed',
                        action='store', type=int, default=4,
                        help='Seed for random number generator.')
    parser.add_argument('--image-resize', dest='image_resize',
                        action='store', type=float, default=1.0,
                        help='Resize factor for all images.')
    parser.add_argument('--num-queries', dest='num_queries',
                        action='store', type=int, default=20,
                        help='Number of object queries, ie detection slot, in a frame.')
    parser.add_argument('--max-frames', dest='max_frames',
                        action='store', type=int, default=0,
                        help='Maximum number of frames to use from each videos. Default is 0, which uses all frames.')
    parser.add_argument('--batch-size', dest='batch_size',
                        action='store', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--log-level', dest='log_level',
                        action='store', type=str, default='INFO',
                        help='Logging level.', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--saved-model-path', dest='saved_model_path',
                        action='store', type=str, default=SAVED_MODEL_PATH,
                        help='Path to model parameters to load.')

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
