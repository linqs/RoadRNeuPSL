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
from models.analysis import save_images_with_bounding_boxes
from models.analysis import ratio_to_pixel_coordinates
from models.evaluation import count_violated_constraints
from models.evaluation import filter_map_pred_and_truth
from models.evaluation import load_ground_truth_for_detections
from models.evaluation import mean_average_precision
from models.evaluation import precision_recall_f1
from models.trainer import Trainer
from utils import get_torch_device
from utils import load_constraint_file
from utils import load_json_file
from utils import seed_everything
from utils import write_json_file
from utils import BASE_RESULTS_DIR
from utils import BOX_CONFIDENCE_THRESHOLD
from utils import EVALUATION_METRICS_FILENAME
from utils import HARD_CONSTRAINTS_PATH
from utils import IOU_THRESHOLD
from utils import LABEL_CONFIDENCE_THRESHOLD
from utils import NEURAL_TEST_INFERENCE_DIR
from utils import NEURAL_TRAINED_MODEL_DIR
from utils import NEURAL_TRAINED_MODEL_FILENAME
from utils import NEURAL_VALID_INFERENCE_DIR
from utils import NUM_SAVED_IMAGES
from utils import PREDICTION_LABELS_JSON_FILENAME
from utils import PREDICTIONS_JSON_FILENAME
from utils import SEED
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import VIDEO_PARTITIONS


def evaluate_dataset(dataset, arguments):
    if os.path.isfile(os.path.join(arguments.output_dir, PREDICTIONS_JSON_FILENAME)):
        logging.info("Skipping neural evaluation for %s, already exists." % (arguments.output_dir,))
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

    frame_indexes, boxes, logits, _ = trainer.eval(dataloader, calculate_loss=False, keep_predictions=True)

    write_json_file(os.path.join(arguments.output_dir, PREDICTIONS_JSON_FILENAME), {"frame_indexes": frame_indexes, "logits": logits, "boxes": boxes})


def calculate_metrics(dataset, output_dir):
    logging.info("Loading predicted probabilities and labels.")
    predictions = load_json_file(os.path.join(output_dir, PREDICTIONS_JSON_FILENAME))

    logging.info("Calculating metrics.")

    logging.info("Calculating mean average precision at iou threshold of {}.".format(IOU_THRESHOLD))
    filtered_pred, filtered_truth = filter_map_pred_and_truth(dataset,
                                                              torch.Tensor(predictions["frame_indexes"]),
                                                              torch.Tensor(predictions["box_predictions"]),
                                                              torch.Tensor(predictions["class_predictions"]),
                                                              IOU_THRESHOLD, LABEL_CONFIDENCE_THRESHOLD)
    mean_avg_prec = mean_average_precision(filtered_pred, filtered_truth, IOU_THRESHOLD)
    logging.info("Mean average precision: %s" % mean_avg_prec)

    logging.info("Calculating precision, recall, and f1 at iou threshold of {}.".format(IOU_THRESHOLD))
    precision, recall, f1 = precision_recall_f1(dataset, predictions["class_predictions"], IOU_THRESHOLD, LABEL_CONFIDENCE_THRESHOLD)
    logging.info("Precision: %s" % precision)
    logging.info("Recall: %s" % recall)
    logging.info("F1: %s" % f1)

    logging.info("Counting constraint violations.")
    violations = count_violated_constraints(predictions["class_predictions"], load_constraint_file(HARD_CONSTRAINTS_PATH))
    logging.info("Number of constraint violations: {}".format(violations[0]))
    logging.info("Number of frames with constraint violations: {}".format(violations[1]))
    logging.info("Constraint violation dict: {}".format(violations[2]))

    metrics = {
        "mean_avg_prec": mean_avg_prec,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_constraint_violations": num_constraint_violations,
        "num_frames_with_violation": num_frames_with_violation,
        "constraint_violation_dict": constraint_violation_dict
    }

    logging.info("Saving metrics to %s" % os.path.join(output_dir, EVALUATION_METRICS_FILENAME))
    write_json_file(os.path.join(output_dir, EVALUATION_METRICS_FILENAME), metrics)


def save_images(dataset, arguments):
    if arguments.save_images.upper() == "NONE":
        pass
    elif arguments.save_images.upper() == "BOXES":
        save_images_with_bounding_boxes(dataset, arguments.output_dir, False,
                                        arguments.max_saved_images,
                                        LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD,
                                        write_ground_truth=(not arguments.test_evaluation),
                                        test=arguments.test_evaluation)
    elif arguments.save_images.upper() == "BOXES_AND_LABELS":
        save_images_with_bounding_boxes(dataset, arguments.output_dir, True,
                                        arguments.max_saved_images,
                                        LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD,
                                        write_ground_truth=(not arguments.test_evaluation),
                                        test=arguments.test_evaluation)
    else:
        raise ValueError("Invalid save_images argument: %s" % arguments.save_images)


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

    logging.info("Evaluating dataset.")
    evaluate_dataset(dataset, arguments)

    logging.info("Calculating metrics.")
    if not arguments.test_evaluation:
        calculate_metrics(dataset, arguments.output_dir)

    logging.info("Saving images with bounding boxes.")
    save_images(dataset, arguments)


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
