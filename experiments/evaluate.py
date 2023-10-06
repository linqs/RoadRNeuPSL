import argparse
import logging
import os
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger
import utils

from torch.utils.data import DataLoader

from data.stream_roadr_dataset import RoadRDataset
from experiments.pretrain import build_model
from models.analysis import save_images_with_bounding_boxes
from models.evaluation import count_violated_pairwise_constraints
from models.evaluation import filter_detections
from models.evaluation import format_pairwise_constraints
from models.evaluation import load_ground_truth_for_detections
from models.evaluation import mean_average_precision
from models.trainer import Trainer
from utils import BASE_RESULTS_DIR
from utils import BOX_CONFIDENCE_THRESHOLD
from utils import CONSTRAINTS_PATH
from utils import EVALUATION_METRICS_FILENAME
from utils import PREDICTION_LOGITS_JSON_FILENAME
from utils import PREDICTION_LOGITS_PKL_FILENAME
from utils import PREDICTION_LABELS_JSON_FILENAME
from utils import PREDICTION_LABELS_PKL_FILENAME
from utils import IOU_THRESHOLD
from utils import LABEL_CONFIDENCE_THRESHOLD
from utils import NUM_SAVED_IMAGES
from utils import SEED
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import TRAINED_MODEL_DIR
from utils import TRAINED_MODEL_FILENAME
from utils import VIDEO_PARTITIONS


def sigmoid_list_of_logits(logits):
    return torch.nn.Sigmoid()(torch.Tensor(logits)).tolist()


def sort_by_confidence(frame_logits, frame_boxes):
    return zip(*sorted(zip(frame_logits, frame_boxes), key=lambda x: x[0][-1], reverse=True))


def save_logits_and_labels(dataset, frame_indexes, logits, boxes, output_dir, from_logits=True):
    # TODO(Charles): rescale bounding boxes to original image size.
    logits_output_dict = {}
    labels_output_dict = {}

    for frame_index, frame_logits, frame_boxes in zip(frame_indexes, logits, boxes):
        frame_id = dataset.get_frame_id(frame_index)

        if frame_id[0] not in logits_output_dict:
            logits_output_dict[frame_id[0]] = {}
            labels_output_dict[frame_id[0]] = {}

        frame_logits, frame_boxes = sort_by_confidence(frame_logits, frame_boxes)

        logits_output_dict[frame_id[0]][frame_id[1]] = []
        labels_output_dict[frame_id[0]][frame_id[1]] = []
        for logits, box in zip(frame_logits, frame_boxes):
            if from_logits:
                prediction = sigmoid_list_of_logits(logits)
            else:
                prediction = logits

            if prediction[-1] >= BOX_CONFIDENCE_THRESHOLD:
                logits_output_dict[frame_id[0]][frame_id[1]].append({"labels": prediction[:-1], "bbox": box})

                predicted_labels = []
                for i in range(len(prediction) - 1):
                    if prediction[i] > LABEL_CONFIDENCE_THRESHOLD:
                        predicted_labels.append(i)

                labels_output_dict[frame_id[0]][frame_id[1]].append({"labels": predicted_labels, "bbox": box})
            else:
                # TODO(Charles): Should we even include boxes with low confidence in submission?
                logits_output_dict[frame_id[0]][frame_id[1]].append({"labels": [0.0] * len(prediction[:-1]), "bbox": box})
                labels_output_dict[frame_id[0]][frame_id[1]].append({"labels": [], "bbox": box})

    logging.info("Saving pkl prediction logits to %s" % os.path.join(output_dir, PREDICTION_LOGITS_PKL_FILENAME))
    utils.write_pkl_file(os.path.join(output_dir, PREDICTION_LOGITS_PKL_FILENAME), logits_output_dict)

    logging.info("Saving json prediction logits to %s" % os.path.join(output_dir, PREDICTION_LOGITS_JSON_FILENAME))
    utils.write_json_file(os.path.join(output_dir, PREDICTION_LOGITS_JSON_FILENAME), logits_output_dict)

    logging.info("Saving pkl prediction labels to %s" % os.path.join(output_dir, PREDICTION_LABELS_PKL_FILENAME))
    utils.write_pkl_file(os.path.join(output_dir, PREDICTION_LABELS_PKL_FILENAME), labels_output_dict)

    logging.info("Saving json prediction labels to %s" % os.path.join(output_dir, PREDICTION_LABELS_JSON_FILENAME))
    utils.write_json_file(os.path.join(output_dir, PREDICTION_LABELS_JSON_FILENAME), labels_output_dict)


def format_saved_predictions(predictions, dataset):
    frame_indexes, box_predictions, class_predictions = [], [], []

    for video_index, video_predictions in predictions.items():
        for frame_index, frame_predictions in video_predictions.items():
            box_predictions.append([])
            class_predictions.append([])

            frame_indexes.append(dataset.get_frame_index((video_index, frame_index)))
            for frame_prediction in frame_predictions:
                box_predictions[-1].append(frame_prediction["bbox"])
                class_predictions[-1].append(frame_prediction["labels"])

    return frame_indexes, class_predictions, box_predictions


def evaluate_dataset(dataset, arguments):
    if os.path.isfile(os.path.join(arguments.output_dir, PREDICTION_LABELS_JSON_FILENAME)) and \
            os.path.isfile(os.path.join(arguments.output_dir, PREDICTION_LABELS_PKL_FILENAME)):
        logging.info("Skipping evaluation for %s, already exists." % (arguments.output_dir,))
        return

    os.makedirs(os.path.join(arguments.output_dir), exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=arguments.batch_size,
                            shuffle=False, num_workers=int(os.cpu_count()) - 2,
                            prefetch_factor=4, persistent_workers=True)

    logging.info("Building and loading pre-trained model.")
    model = build_model()
    model.load_state_dict(torch.load(arguments.saved_model_path))

    logging.info("Evaluating model.")
    trainer = Trainer(model, None, None, utils.get_torch_device(), os.path.join(arguments.output_dir))

    frame_indexes, boxes, logits = trainer.evaluate(dataloader)

    save_logits_and_labels(dataset, frame_indexes, logits, boxes, output_dir=arguments.output_dir)


def calculate_metrics(dataset, output_dir):
    if os.path.isfile(os.path.join(output_dir, EVALUATION_METRICS_FILENAME)):
        logging.info("Skipping calculation metrics for %s, already exists." % (os.path.join(output_dir, EVALUATION_METRICS_FILENAME),))

        results = utils.load_json_file(os.path.join(output_dir, EVALUATION_METRICS_FILENAME))
        logging.info("Saved metrics: %s" % results)
        return

    logging.info("Loading predictions.")
    predictions = utils.load_json_file(os.path.join(output_dir, PREDICTION_LOGITS_JSON_FILENAME))
    frame_indexes, class_predictions, box_predictions = format_saved_predictions(predictions, dataset)

    logging.info("Calculating metrics.")

    logging.info("Calculating mean average precision.")
    filtered_detections, filtered_detection_indexes = filter_detections(torch.Tensor(frame_indexes), torch.Tensor(box_predictions), torch.Tensor(class_predictions), IOU_THRESHOLD, LABEL_CONFIDENCE_THRESHOLD)
    filtered_detections_ground_truth = load_ground_truth_for_detections(dataset, filtered_detection_indexes)
    mean_avg_prec = mean_average_precision(filtered_detections_ground_truth, filtered_detections, IOU_THRESHOLD)
    logging.info("Mean average precision: %s" % mean_avg_prec)

    logging.info("Counting constraint violations.")
    pairwise_constraints = format_pairwise_constraints(utils.load_csv_file(CONSTRAINTS_PATH, ','))
    num_constraint_violations, num_frames_with_violation, constraint_violation_dict = count_violated_pairwise_constraints(class_predictions, pairwise_constraints, positive_threshold=LABEL_CONFIDENCE_THRESHOLD)
    logging.info("Number of constraint violations: {}".format(num_constraint_violations))
    logging.info("Number of frames with constraint violations: {}".format(num_frames_with_violation))
    logging.info("Constraint violation dict: {}".format(constraint_violation_dict))

    metrics = {
        "mean_avg_prec": mean_avg_prec,
        "num_constraint_violations": num_constraint_violations,
        "num_frames_with_violation": num_frames_with_violation,
        "constraint_violation_dict": constraint_violation_dict
    }

    logging.info("Saving metrics to %s" % os.path.join(output_dir, EVALUATION_METRICS_FILENAME))
    utils.write_json_file(os.path.join(output_dir, EVALUATION_METRICS_FILENAME), metrics)


def save_images(dataset, arguments):
    if arguments.save_images.upper() == "NONE":
        pass
    elif arguments.save_images.upper() == "BOXES":
        save_images_with_bounding_boxes(dataset, arguments.output_dir, False, arguments.max_saved_images, LABEL_CONFIDENCE_THRESHOLD)
    elif arguments.save_images.upper() == "BOXES_AND_LABELS":
        save_images_with_bounding_boxes(dataset, arguments.output_dir, True, arguments.max_saved_images, LABEL_CONFIDENCE_THRESHOLD)
    else:
        raise ValueError("Invalid save_images argument: %s" % arguments.save_images)


def main(arguments):
    utils.seed_everything(arguments.seed)

    logger.initLogging(arguments.log_level)
    logging.info("Beginning evaluating.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    logging.info("Loading evaluation videos: %s" % arguments.eval_videos.upper())
    dataset = RoadRDataset(VIDEO_PARTITIONS[arguments.task][arguments.eval_videos.upper()], TRAIN_VALIDATION_DATA_PATH, arguments.image_resize, max_frames=arguments.max_frames)

    logging.info("Evaluating dataset.")
    evaluate_dataset(dataset, arguments)

    logging.info("Calculating metrics.")
    calculate_metrics(dataset, arguments.output_dir)

    logging.info("Saving images with bounding boxes.")
    save_images(dataset, arguments)


def _load_args():
    parser = argparse.ArgumentParser(description="Evaluating Model.")

    parser.add_argument("--task", dest="task", type=str, choices=["task1", "task2"])
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
                        help="Videos to evaluate on.", choices=["TRAIN", "VALID"])
    parser.add_argument("--batch-size", dest="batch_size",
                        action="store", type=int, default=8,
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

    if arguments.saved_model_path is None:
        arguments.saved_model_path = os.path.join(BASE_RESULTS_DIR, arguments.task, TRAINED_MODEL_DIR, TRAINED_MODEL_FILENAME)

    if arguments.output_dir is None:
        arguments.output_dir = os.path.join(BASE_RESULTS_DIR, arguments.task, TRAINED_MODEL_DIR, "evaluation")

    return arguments


if __name__ == "__main__":
    main(_load_args())
