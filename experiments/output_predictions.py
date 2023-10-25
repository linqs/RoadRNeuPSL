import argparse
import logging
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger

from utils import load_json_file
from utils import ratio_to_pixel_coordinates
from utils import BOX_CONFIDENCE_THRESHOLD
from utils import IMAGE_HEIGHT
from utils import LABEL_CONFIDENCE_THRESHOLD
from utils import LABEL_MAPPING
from utils import IMAGE_WIDTH


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

            print("Box: %s" % (box,))
            print("Labels: %s" % (labels,))
            if arguments.show_probabilities:
                print("Probabilities: %s" % (class_preds[:-1],))


def main(arguments):
    logger.initLogging(arguments.log_level)

    logging.info("Loading predictions.")
    predictions = load_json_file(arguments.predictions_path)

    logging.info("Formatting predictions.")
    submission_predictions = output_prediction(arguments, predictions)


def _load_args():
    parser = argparse.ArgumentParser(description="Outputting Predictions for Frame in Video.")

    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default="INFO",
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--image-resize", dest="image_resize",
                        action="store", type=float, default=1.0,
                        help="Resize factor for all images.")
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

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
