import argparse
import logging
import os
import sys

import torch

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger

from data.roadr_dataset import RoadRDataset
from experiments.pretrain import build_model
from models.trainer import Trainer
from utils import get_torch_device
from utils import seed_everything
from utils import write_json_file
from utils import BASE_RESULTS_DIR
from utils import NEURAL_TEST_INFERENCE_DIR
from utils import NEURAL_TRAINED_MODEL_DIR
from utils import NEURAL_TRAINED_MODEL_FILENAME
from utils import NEURAL_VALID_INFERENCE_DIR
from utils import NUM_SAVED_IMAGES
from utils import PREDICTIONS_JSON_FILENAME
from utils import SEED
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import VIDEO_PARTITIONS


def sort_detections_in_frames(pred_labels, pred_boxes):
    """
    Sorts the detections in each frame by their confidence score.
    :param pred_labels: List of lists of predicted labels.
    :param pred_boxes: List of lists of predicted boxes.
    :return: Sorted lists of predicted labels and boxes.
    """
    sorted_boxes = []
    sorted_labels = []

    for frame_pred_labels, frame_pred_boxes in zip(pred_labels, pred_boxes):
        frame_pred_labels, frame_pred_boxes = zip(*sorted(zip(frame_pred_labels, frame_pred_boxes), key=lambda x: x[0][-1], reverse=True))

        sorted_boxes.append(frame_pred_boxes)
        sorted_labels.append(frame_pred_labels)

    return sorted_boxes, sorted_labels


def save_predictions(predictions, dataset, arguments):
    """
    Sorts and saves the predictions to a json file.
    :param predictions: Dictionary of predictions containing frame indexes, boxes, and logits.
    :param dataset: Dataset used to get frame ids.
    :param arguments: Arguments used to get output directory.
    :return:
    """
    frame_indexes, boxes, logits, _ = predictions

    sorted_boxes, sorted_logits = sort_detections_in_frames(logits, boxes)
    class_predictions = torch.sigmoid(torch.tensor(sorted_logits)).tolist()
    frame_ids = [dataset.get_frame_id(frame_index) for frame_index in frame_indexes]

    write_json_file(os.path.join(arguments.output_dir, PREDICTIONS_JSON_FILENAME), {"frame_ids": frame_ids, "frame_indexes": frame_indexes, "box_predictions": sorted_boxes, "class_predictions": class_predictions}, indent=None)


def predict_dataset(dataset, arguments):
    if os.path.isfile(os.path.join(arguments.output_dir, PREDICTIONS_JSON_FILENAME)):
        logging.info("Skipping predicting neural for %s, already exists." % (arguments.output_dir,))
        return

    os.makedirs(os.path.join(arguments.output_dir), exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=arguments.batch_size,
                            shuffle=False, num_workers=int(os.cpu_count()) - 2,
                            prefetch_factor=4, persistent_workers=True)

    logging.info("Building and loading pre-trained model from {}.".format(arguments.saved_model_path))
    model = build_model()
    model.load_state_dict(torch.load(arguments.saved_model_path, map_location=get_torch_device()))

    logging.info("Running neural inference with trained model {}.".format(arguments.saved_model_path))
    trainer = Trainer(model, None, None, get_torch_device(), os.path.join(arguments.output_dir))

    return trainer.eval(dataloader, calculate_loss=False, keep_predictions=True)


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

    logging.info("Predicting dataset.")
    predictions = predict_dataset(dataset, arguments)

    logging.info("Saving predictions.")
    save_predictions(predictions, dataset, arguments)


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
    parser.add_argument("--batch-size", dest="batch_size",
                        action="store", type=int, default=4,
                        help="Batch size.")
    parser.add_argument("--saved-model-path", dest="saved_model_path",
                        action="store", type=str, default=None,
                        help="Path to model parameters to load.")
    parser.add_argument("--output-dir", dest="output_dir",
                        action="store", type=str, default=None,
                        help="Directory to save results to.")
    parser.add_argument("--save-images", dest="save_images",
                        action="store", type=str, default="BOXES_AND_LABELS",
                        help="Save images with bounding boxes.", choices=["NONE", "BOXES", "BOXES_AND_LABELS"])
    parser.add_argument("--max-saved-images", dest="max_saved_images",
                        action="store", type=int, default=NUM_SAVED_IMAGES,
                        help="Maximum number of images saved per video.")

    arguments = parser.parse_args()

    arguments.test_evaluation = arguments.eval_videos.upper() == "TEST"

    if arguments.saved_model_path is None:
        arguments.saved_model_path = os.path.join(BASE_RESULTS_DIR, arguments.task, NEURAL_TRAINED_MODEL_DIR, NEURAL_TRAINED_MODEL_FILENAME)

    if arguments.output_dir is None:
        if arguments.test_evaluation:
            arguments.output_dir = os.path.join(BASE_RESULTS_DIR, arguments.task, NEURAL_TEST_INFERENCE_DIR, "evaluation")
        else:
            arguments.output_dir = os.path.join(BASE_RESULTS_DIR, arguments.task, NEURAL_VALID_INFERENCE_DIR, "evaluation")

    return arguments


if __name__ == "__main__":
    main(_load_args())