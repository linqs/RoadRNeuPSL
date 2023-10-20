import argparse
import logging
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger

from data.roadr_dataset import RoadRDataset
from models.evaluation import save_images_with_bounding_boxes, agent_nms
from models.evaluation import count_violated_constraints
from models.evaluation import filter_map_pred_and_truth
from models.evaluation import mean_average_precision
from models.evaluation import precision_recall_f1
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
from utils import NUM_CLASSES
from utils import NUM_QUERIES
from utils import NUM_SAVED_IMAGES
from utils import PREDICTIONS_JSON_FILENAME
from utils import SEED
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import VIDEO_PARTITIONS


def filter_predictions(dataset, predictions, arguments):
    logging.info("Filtering predictions.")
    nms_keep_indices = agent_nms(
        np.array(self.all_box_predictions).reshape((len(self.dataset), NUM_QUERIES, 4)),
        np.array(self.all_class_predictions)[:, :, -1].reshape((len(self.dataset), NUM_QUERIES)),
        np.array(self.all_class_predictions).reshape((len(self.dataset), NUM_QUERIES, NUM_CLASSES + 1))[:, :, :-1],
    )

    new_all_box_predictions = torch.zeros(size=(len(self.dataset), NUM_QUERIES, 4), dtype=torch.float32)
    new_all_class_predictions = torch.zeros(size=(len(self.dataset), NUM_QUERIES, NUM_CLASSES + 1), dtype=torch.float32)

    # Construct the new prediction by keeping only the nms_keep_indices and padding with 0s otherwise.
    for frame_index in range(len(self.all_frame_indexes)):
        for box_index in range(NUM_QUERIES):
            if box_index in nms_keep_indices[frame_index]:
                new_all_box_predictions[frame_index][box_index] = torch.tensor(self.all_box_predictions[frame_index][box_index])
                new_all_class_predictions[frame_index][box_index] = torch.tensor(self.all_class_predictions[frame_index][box_index])

    sorted_new_all_box_predictions = torch.zeros(size=(len(self.dataset), NUM_QUERIES, 4), dtype=torch.float32)
    sorted_new_all_class_predictions = torch.zeros(size=(len(self.dataset), NUM_QUERIES, NUM_CLASSES + 1), dtype=torch.float32)

    # Sort boxes by confidence.
    sorted_indexes = torch.argsort(new_all_class_predictions[:, :, -1], descending=True)

    for frame_index in range(len(self.all_frame_indexes)):
        for box_index in range(NUM_QUERIES):
            sorted_new_all_box_predictions[frame_index][box_index] = new_all_box_predictions[frame_index][sorted_indexes[frame_index][box_index]]
            sorted_new_all_class_predictions[frame_index][box_index] = new_all_class_predictions[frame_index][sorted_indexes[frame_index][box_index]]

    sorted_new_all_box_predictions.reshape((len(self.dataset), NUM_QUERIES * 4))
    sorted_new_all_class_predictions.reshape((len(self.dataset), NUM_QUERIES * (NUM_CLASSES + 1)))
    return predictions


def calculate_metrics(dataset, predictions, output_dir):
    logging.info("Calculating metrics.")

    logging.info("Calculating mean average precision at iou threshold of {}.".format(IOU_THRESHOLD))
    filtered_pred, filtered_truth = filter_map_pred_and_truth(dataset,
                                                              predictions["frame_indexes"],
                                                              torch.Tensor(predictions["box_predictions"]),
                                                              torch.Tensor(predictions["class_predictions"]),
                                                              IOU_THRESHOLD, LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD)
    mean_avg_prec = mean_average_precision(filtered_pred, filtered_truth, IOU_THRESHOLD)
    logging.info("Mean average precision: %s" % mean_avg_prec)

    logging.info("Calculating precision, recall, and f1 at iou threshold of {}.".format(IOU_THRESHOLD))
    precision, recall, f1 = precision_recall_f1(dataset,
                                                predictions["frame_indexes"],
                                                predictions["box_predictions"],
                                                predictions["class_predictions"],
                                                IOU_THRESHOLD, LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD)
    logging.info("Precision: %s" % precision)
    logging.info("Recall: %s" % recall)
    logging.info("F1: %s" % f1)

    logging.info("Counting constraint violations.")
    violations = count_violated_constraints(load_constraint_file(HARD_CONSTRAINTS_PATH),
                                            predictions["frame_indexes"],
                                            predictions["class_predictions"],
                                            LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD)
    logging.info("Constraint violations: {}".format(violations[0]))
    logging.info("Frames with constraint violations: {}".format(violations[1]))
    logging.info("Constraint violation dict: {}".format(violations[2]))

    metrics = {
        "mean_avg_prec": mean_avg_prec,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "constraint_violations": violations[0],
        "frames_with_violations": violations[1],
        "constraint_violation_dict": violations[2]
    }

    logging.info("Saving metrics to %s" % os.path.join(output_dir, EVALUATION_METRICS_FILENAME))
    write_json_file(os.path.join(output_dir, EVALUATION_METRICS_FILENAME), metrics)


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

    logging.info("Loading predicted probabilities and labels.")
    predictions = load_json_file(os.path.join(arguments.output_dir, PREDICTIONS_JSON_FILENAME))

    if arguments.filter_predictions:
        predictions = filter_predictions(dataset, predictions, arguments)

    if not arguments.test_evaluation:
        calculate_metrics(dataset, predictions, arguments.output_dir)

    logging.info("Saving images with bounding boxes.")
    save_images_with_bounding_boxes(dataset, arguments.output_dir, arguments.max_saved_images,
                                    LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD,
                                    test=arguments.test_evaluation)


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
    parser.add_argument("--filter-predictions", dest="filter_predictions",
                        action="store_true", default=False,
                        help="Turn on filtering predictions.")

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
