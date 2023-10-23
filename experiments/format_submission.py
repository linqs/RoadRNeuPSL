import argparse
import logging
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger

from data.roadr_dataset import RoadRDataset
from experiments.evaluate import filter_predictions
from utils import load_json_file
from utils import ratio_to_pixel_coordinates
from utils import seed_everything
from utils import write_pkl_file
from utils import BASE_RESULTS_DIR
from utils import BOX_CONFIDENCE_THRESHOLD
from utils import LABEL_CONFIDENCE_THRESHOLD
from utils import NEURAL_TEST_INFERENCE_DIR
from utils import NEURAL_VALID_INFERENCE_DIR
from utils import PREDICTIONS_JSON_FILENAME
from utils import SEED
from utils import SUBMISSION_PREDICTIONS_PKL_FILENAME
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import VIDEO_PARTITIONS


def format_predictions(dataset, predictions, task):
    formatted_predicitons = {}
    frame_ids, box_predictions, class_predictions = predictions["frame_ids"], predictions["box_predictions"], predictions["class_predictions"]

    for frame_id, frame_box_predictions, frame_class_predictions in zip(frame_ids, box_predictions, class_predictions):
        if frame_id[0] not in formatted_predicitons:
            formatted_predicitons[frame_id[0]] = {}

        scaled_frame_box_predictions = ratio_to_pixel_coordinates(torch.tensor(frame_box_predictions), dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize).tolist()

        formatted_predicitons[frame_id[0]][frame_id[1]] = []
        for box_prediction, box_class_predictions in zip(scaled_frame_box_predictions, frame_class_predictions):
            if box_class_predictions[-1] < BOX_CONFIDENCE_THRESHOLD:
                break

            if task == "task1":
                formatted_predicitons[frame_id[0]][frame_id[1]].append({"labels": box_class_predictions[:-1], "bbox": box_prediction})
            elif task == "task2":
                formatted_predicitons[frame_id[0]][frame_id[1]].append({"labels": [index for index, label in enumerate(box_class_predictions[:-1]) if label > LABEL_CONFIDENCE_THRESHOLD], "bbox": box_prediction})

    return formatted_predicitons


def main(arguments):
    seed_everything(arguments.seed)

    logger.initLogging(arguments.log_level)
    logging.info("Beginning evaluating.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    logging.info("Loading evaluation videos: %s" % arguments.eval_videos.upper())
    annotations_path = TRAIN_VALIDATION_DATA_PATH if not arguments.test_evaluation else None
    dataset = RoadRDataset(VIDEO_PARTITIONS[arguments.task][arguments.eval_videos.upper()], annotations_path, arguments.image_resize,
                           max_frames=arguments.max_frames, test=arguments.test_evaluation)

    logging.info("Loading predictions.")
    predictions = load_json_file(os.path.join(arguments.output_dir, PREDICTIONS_JSON_FILENAME))

    if arguments.filter_predictions:
        predictions = filter_predictions(dataset, predictions)

    logging.info("Formatting predictions.")
    submission_predictions = format_predictions(dataset, predictions, arguments.task)

    logging.info("Saving submission pkl file to %s" % os.path.join(arguments.output_dir, SUBMISSION_PREDICTIONS_PKL_FILENAME))
    write_pkl_file(os.path.join(arguments.output_dir, SUBMISSION_PREDICTIONS_PKL_FILENAME), submission_predictions)

    logging.info("Zipping submission pkl file to %s" % os.path.join(arguments.output_dir, arguments.task + "_" + arguments.eval_videos.lower() + ".zip"))
    os.system("cd %s; zip %s %s" % (arguments.output_dir, arguments.task + "_" + arguments.eval_videos.lower() + ".zip", SUBMISSION_PREDICTIONS_PKL_FILENAME))


def _load_args():
    parser = argparse.ArgumentParser(description="Evaluating Model.")

    parser.add_argument("--task", dest="task",
                        type=str, choices=["task1", "task2"])
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
    parser.add_argument("--eval-videos", dest="eval_videos",
                        action="store", type=str, default="VALID",
                        help="Videos to evaluate on.", choices=["TRAIN", "VALID", "TEST"])
    parser.add_argument("--output-dir", dest="output_dir",
                        action="store", type=str, default=None,
                        help="Directory to save results to.")
    parser.add_argument("--filter-predictions", dest="filter_predictions",
                        action="store_true", default=False,
                        help="Turn on filtering predictions.")

    arguments = parser.parse_args()

    arguments.test_evaluation = arguments.eval_videos.upper() == "TEST"

    if arguments.output_dir is None:
        if arguments.test_evaluation:
            arguments.output_dir = os.path.join(BASE_RESULTS_DIR, arguments.task, NEURAL_TEST_INFERENCE_DIR, "evaluation")
        else:
            arguments.output_dir = os.path.join(BASE_RESULTS_DIR, arguments.task, NEURAL_VALID_INFERENCE_DIR, "evaluation")

    return arguments


if __name__ == "__main__":
    main(_load_args())
