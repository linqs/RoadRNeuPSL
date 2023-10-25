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
from utils import write_json_file
from utils import BOX_CONFIDENCE_THRESHOLD
from utils import IMAGE_HEIGHT
from utils import IMAGE_WIDTH
from utils import LABEL_CONFIDENCE_THRESHOLD
from utils import LABEL_MAPPING
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import VIDEO_PARTITIONS


def output_prediction(arguments, predictions):
    for frame_id, box_prediction, class_prediction in zip(predictions["frame_ids"], predictions["box_predictions"], predictions["class_predictions"]):
        if not frame_id[0] == arguments.video_name or not frame_id[1] == arguments.frame_name:
            continue

        print("Frame: %s" % (frame_id,))
        box_prediction = ratio_to_pixel_coordinates(torch.tensor(box_prediction), IMAGE_HEIGHT / arguments.image_resize, IMAGE_WIDTH / arguments.image_resize).tolist()

        for box, class_preds in zip(box_prediction, class_prediction):
            if class_preds[-1] < arguments.box_threshold:
                continue

            labels = [LABEL_MAPPING[index] for index, label in enumerate(class_preds[:-1]) if label > arguments.label_threshold]

            print("Box: %s Box Confidence: %f" % (box, class_preds[-1]))
            print("Labels: %s" % (labels,))
            if arguments.show_probabilities:
                print("Probabilities: %s" % (class_preds[:-1],))


def write_predictions_subset(arguments, predictions, output_path):
    predictions_subset = {"frame_ids": [], "box_predictions": [], "class_predictions": []}
    predictions_subset_count = {}
    for frame_id, box_prediction, class_prediction in zip(predictions["frame_ids"], predictions["box_predictions"], predictions["class_predictions"]):
        if frame_id[0] not in predictions_subset_count:
            predictions_subset_count[frame_id[0]] = 0

        if predictions_subset_count[frame_id[0]] >= arguments.predictions_subset_size:
            continue

        predictions_subset["frame_ids"].append(frame_id)
        predictions_subset["box_predictions"].append(box_prediction)
        predictions_subset["class_predictions"].append(class_prediction)

        predictions_subset_count[frame_id[0]] += 1

    logging.info("Saving predictions subset to %s." % (output_path,))
    write_json_file(output_path, predictions_subset, indent=None)


def main(arguments):
    logger.initLogging(arguments.log_level)

    logging.info("Loading evaluation videos: %s" % arguments.eval_videos.upper())
    annotations_path = TRAIN_VALIDATION_DATA_PATH if not arguments.test_evaluation else None
    dataset = RoadRDataset(VIDEO_PARTITIONS[arguments.task][arguments.eval_videos.upper()], annotations_path, arguments.image_resize,
                           max_frames=arguments.max_frames, test=arguments.test_evaluation)

    if not os.path.isfile(arguments.predictions_path[:-5] + "-subset.json"):
        logging.info("Predictions subset not found, loading full predictions.")
        predictions = load_json_file(arguments.predictions_path)
        write_predictions_subset(arguments, predictions, arguments.predictions_path[:-5] + "-subset.json")

    logging.info("Loading predictions subset.")
    predictions = load_json_file(arguments.predictions_path[:-5] + "-subset.json")

    if arguments.filter_predictions:
        predictions = filter_predictions(dataset, predictions)

    logging.info("Formatting predictions.")
    output_prediction(arguments, predictions)


def _load_args():
    parser = argparse.ArgumentParser(description="Outputting Predictions for Frame in Video.")

    parser.add_argument("--task", dest="task",
                        type=str, choices=["task1", "task2"])
    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default="INFO",
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--eval-videos", dest="eval_videos",
                        action="store", type=str, default="VALID",
                        help="Videos to evaluate on.", choices=["TRAIN", "VALID", "TEST"])
    parser.add_argument("--image-resize", dest="image_resize",
                        action="store", type=float, default=1.0,
                        help="Resize factor for all images.")
    parser.add_argument("--max-frames", dest="max_frames",
                        action="store", type=int, default=0,
                        help="Maximum number of frames to use from each videos. Default is 0, which uses all frames.")
    parser.add_argument("--predictions-path", dest="predictions_path",
                        action="store", type=str,
                        help="Resize factor for all images.")
    parser.add_argument("--video-name", dest="video_name",
                        action="store", type=str,
                        help="Name of video to print predictions.")
    parser.add_argument("--frame-name", dest="frame_name",
                        action="store", type=str,
                        help="Name of frame to print predictions.")
    parser.add_argument("--box-threshold", dest="box_threshold",
                        action="store", type=float, default=BOX_CONFIDENCE_THRESHOLD,
                        help="Threshold for box confidence.")
    parser.add_argument("--label-threshold", dest="label_threshold",
                        action="store", type=float, default=LABEL_CONFIDENCE_THRESHOLD,
                        help="Threshold for label confidence.")
    parser.add_argument("--show-probabilities", dest="show_probabilities",
                        action="store_true", default=False,
                        help="Show probabilities for each label.")
    parser.add_argument("--predictions-subset-size", dest="predictions_subset_size",
                        action="store", type=int, default=10,
                        help="Subset size used for loading quicker.")
    parser.add_argument("--filter-predictions", dest="filter_predictions",
                        action="store_true", default=False,
                        help="Turn on filtering predictions.")

    arguments = parser.parse_args()

    arguments.test_evaluation = arguments.eval_videos.upper() == "TEST"

    return arguments


if __name__ == "__main__":
    main(_load_args())
