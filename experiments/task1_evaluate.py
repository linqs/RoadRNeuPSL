import argparse
import logging
import os
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger
import utils

from torch.utils.data import DataLoader

from data.roadr_dataset import RoadRDataset
from experiments.task1_pretrain import task_1_model
from experiments.task1_pretrain import TASK_NAME
from models.analysis import save_images_with_bounding_boxes
from models.evaluation import filter_detections
from models.evaluation import load_ground_truth_for_detections
from models.evaluation import mean_average_precision
from models.trainer import Trainer
from utils import BASE_RESULTS_DIR
from utils import EVALUATION_METRICS_FILENAME
from utils import EVALUATION_PREDICTION_JSON_FILENAME
from utils import EVALUATION_PREDICTION_PKL_FILENAME
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import TRAINED_MODEL_DIR
from utils import TRAINED_MODEL_FILENAME


EVAL_VIDEOS = {
    "TRAIN": ["2014-07-14-14-49-50_stereo_centre_01",
              "2015-02-03-19-43-11_stereo_centre_04",
              "2015-02-24-12-32-19_stereo_centre_04"],
    "VALID": ["2014-06-26-09-53-12_stereo_centre_02",
              "2014-11-25-09-18-32_stereo_centre_04",
              "2015-02-13-09-16-26_stereo_centre_02"]
}

BOX_CONFIDENCE_THRESHOLD = 0.70
# LABEL_CONFIDENCE_THRESHOLD = 0.20
LABEL_CONFIDENCE_THRESHOLD = 0.025
IOU_THRESHOLD = 0.50


def sigmoid_list_of_logits(logits):
    return torch.nn.Sigmoid()(torch.Tensor(logits)).tolist()


def sort_by_confidence(frame_logits, frame_boxes):
    return zip(*sorted(zip(frame_logits, frame_boxes), key=lambda x: x[0][-1], reverse=True))


def create_task_1_output_format(dataset, frame_indexes, logits, boxes, from_logits=True):
    output_dict = {}

    for frame_index, frame_logits, frame_boxes in zip(frame_indexes, logits, boxes):
        frame_id = dataset.get_frame_and_video_names(frame_index)

        if frame_id[0] not in output_dict:
            output_dict[frame_id[0]] = {}

        frame_logits, frame_boxes = sort_by_confidence(frame_logits, frame_boxes)

        output_dict[frame_id[0]][frame_id[1]] = []
        for logits, box in zip(frame_logits, frame_boxes):
            if from_logits:
                prediction = sigmoid_list_of_logits(logits)
            else:
                prediction = logits

            if prediction[-1] >= BOX_CONFIDENCE_THRESHOLD:
                output_dict[frame_id[0]][frame_id[1]].append({"labels": prediction[:-1], "bbox": box})
            else:
                output_dict[frame_id[0]][frame_id[1]].append({"labels": [0.0] * len(prediction[:-1]), "bbox": box})

    return output_dict


def format_saved_predictions(predicitons, dataset):
    frame_indexes, box_predictions, class_predictions = [], [], []

    for video_index, video_predictions in predicitons.items():
        for frame_index, frame_predictions in video_predictions.items():
            box_predictions.append([])
            class_predictions.append([])

            frame_indexes.append(dataset.video_id_frame_id_to_frame_index[(video_index, frame_index)])
            for frame_prediction in frame_predictions:
                box_predictions[-1].append(frame_prediction["bbox"])
                class_predictions[-1].append(frame_prediction["labels"])

    return frame_indexes, class_predictions, box_predictions


def evaluate_dataset(dataset, arguments):
    if os.path.isfile(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_JSON_FILENAME)) and \
            os.path.isfile(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_PKL_FILENAME)):
        logging.info("Skipping evaluation for %s, already exists." % (arguments.output_dir,))
        return

    os.makedirs(os.path.join(arguments.output_dir), exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False)

    logging.info("Building and loading pre-trained model.")
    model = task_1_model()
    model.load_state_dict(torch.load(arguments.saved_model_path))

    logging.info("Evaluating model.")
    trainer = Trainer(model, None, None, utils.get_torch_device(), os.path.join(arguments.output_dir))

    frame_indexes, boxes, logits = trainer.evaluate(dataloader)
    output_dict = create_task_1_output_format(dataset, frame_indexes, logits, boxes)

    logging.info("Saving pkl predictions to %s" % os.path.join(arguments.output_dir, EVALUATION_PREDICTION_PKL_FILENAME))
    utils.write_pkl_file(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_PKL_FILENAME), output_dict)

    logging.info("Saving json predictions to %s" % os.path.join(arguments.output_dir, EVALUATION_PREDICTION_JSON_FILENAME))
    utils.write_json_file(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_JSON_FILENAME), output_dict)


def calculate_metrics(dataset, arguments):
    if os.path.isfile(os.path.join(arguments.output_dir, EVALUATION_METRICS_FILENAME)):
        logging.info("Skipping calculation metrics for %s, already exists." % (os.path.join(arguments.output_dir, EVALUATION_METRICS_FILENAME),))

        results = utils.load_json_file(os.path.join(arguments.output_dir, EVALUATION_METRICS_FILENAME))
        logging.info("Saved mean average precision: %s" % results["mean_avg_prec"])
        return

    logging.info("Loading predictions.")
    predictions = utils.load_json_file(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_JSON_FILENAME))
    frame_indexes, class_predictions, box_predictions = format_saved_predictions(predictions, dataset)

    logging.info("Calculating metrics.")
    filtered_detections, filtered_detection_indexes = filter_detections(torch.Tensor(frame_indexes), torch.Tensor(box_predictions), torch.Tensor(class_predictions), IOU_THRESHOLD, LABEL_CONFIDENCE_THRESHOLD)
    filtered_detections_ground_truth = load_ground_truth_for_detections(dataset, filtered_detection_indexes)

    mean_avg_prec = mean_average_precision(filtered_detections_ground_truth, filtered_detections, IOU_THRESHOLD)
    logging.info("Mean average precision: %s" % mean_avg_prec)

    logging.info("Saving metrics to %s" % os.path.join(arguments.output_dir, EVALUATION_METRICS_FILENAME))
    utils.write_json_file(os.path.join(arguments.output_dir, EVALUATION_METRICS_FILENAME), {"mean_avg_prec": mean_avg_prec})


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
    logging.info("Beginning evaluating task 1.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    logging.info("Loading evaluation videos: %s" % arguments.eval_videos.upper())
    dataset = RoadRDataset(EVAL_VIDEOS[arguments.eval_videos.upper()], TRAIN_VALIDATION_DATA_PATH, arguments.image_resize, arguments.num_queries, max_frames=arguments.max_frames)

    logging.info("Evaluating dataset.")
    evaluate_dataset(dataset, arguments)

    logging.info("Calculating metrics.")
    calculate_metrics(dataset, arguments)

    logging.info("Saving images with bounding boxes.")
    save_images(dataset, arguments)


def _load_args():
    parser = argparse.ArgumentParser(description="Evaluating Task 1.")

    parser.add_argument("--seed", dest="seed",
                        action="store", type=int, default=4,
                        help="Seed for random number generator.")
    parser.add_argument("--eval-videos", dest="eval_videos",
                        action="store", type=str, default="VALID",
                        help="Videos to evaluate on.", choices=["TRAIN", "VALID"])
    parser.add_argument("--image-resize", dest="image_resize",
                        action="store", type=float, default=1.0,
                        help="Resize factor for all images.")
    parser.add_argument("--num-queries", dest="num_queries",
                        action="store", type=int, default=20,
                        help="Number of object queries, ie detection slot, in a frame.")
    parser.add_argument("--max-frames", dest="max_frames",
                        action="store", type=int, default=0,
                        help="Maximum number of frames to use from each videos. Default is 0, which uses all frames.")
    parser.add_argument("--save-images", dest="save_images",
                        action="store", type=str, default="NONE",
                        help="Save images with bounding boxes.", choices=["NONE", "BOXES", "BOXES_AND_LABELS"])
    parser.add_argument("--max-saved-images", dest="max_saved_images",
                        action="store", type=int, default=10,
                        help="Maximum number of images saved per video.")
    parser.add_argument("--batch-size", dest="batch_size",
                        action="store", type=int, default=2,
                        help="Batch size.")
    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default="INFO",
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--saved-model-path", dest="saved_model_path",
                        action="store", type=str, default=os.path.join(BASE_RESULTS_DIR, TASK_NAME, TRAINED_MODEL_DIR, TRAINED_MODEL_FILENAME),
                        help="Path to model parameters to load.")
    parser.add_argument("--output-dir", dest="output_dir",
                        action="store", type=str, default=os.path.join(BASE_RESULTS_DIR, TASK_NAME, TRAINED_MODEL_DIR, "evaluation"),
                        help="Directory to save results to.")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
