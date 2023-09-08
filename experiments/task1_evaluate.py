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
from experiments.task1_pretrain import TASK_NAME
from experiments.task1_pretrain import task_1_model
from models.trainer import Trainer
from models.evaluation import filter_detections
from models.evaluation import load_labels
from utils import BASE_RESULTS_DIR
from utils import EVALUATION_METRICS_FILENAME
from utils import EVALUATION_PREDICTION_JSON_FILENAME
from utils import EVALUATION_PREDICTION_PKL_FILENAME
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import TRAINED_MODEL_DIR
from utils import TRAINED_MODEL_FILENAME


VALID_VIDEOS = ["2014-06-26-09-53-12_stereo_centre_02",
                "2014-11-25-09-18-32_stereo_centre_04",
                "2015-02-13-09-16-26_stereo_centre_02"]


def create_task_1_output_format(dataset, dataloader, class_predictions, box_predictions):
    output_dict = {}

    all_ids = []
    all_class_labels = []
    all_box_labels = []

    for indexes, frames, labels, boxes in dataloader:
        frames = [dataset.get_frame_and_video_names(index).tolist() for index in indexes]
        all_ids.extend(frames)
        all_class_labels.extend(labels.cpu().tolist())
        all_box_labels.extend(boxes.cpu().tolist())

    for id, class_label, box_label, class_prediction, box_prediction in zip(all_ids, all_class_labels, all_box_labels, class_predictions, box_predictions):
        if id[0] not in output_dict:
            output_dict[id[0]] = {}

        output_dict[id[0]][id[1]] = []
        for prediction, box in zip(class_prediction, box_prediction):
            # if max(prediction[:-1]) > prediction[-1]:
            if prediction[-1] < 0.16:
                output_dict[id[0]][id[1]].append({"labels": prediction[:-1], "bbox": box})

    return output_dict, all_class_labels, all_box_labels


def evaluate_dataset(dataset, arguments):
    if os.path.isfile(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_JSON_FILENAME)) and \
            os.path.isfile(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_PKL_FILENAME)):
        logging.info("Skipping evaluation for %s, already exists." % (arguments.output_dir,))
        return

    os.makedirs(os.path.join(arguments.output_dir), exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False)

    logging.info("Building and loading pre-trained model.")
    model = task_1_model(0.1, arguments.image_resize, arguments.num_queries)
    model.load_state_dict(torch.load(arguments.saved_model_path))

    logging.info("Evaluating model.")
    trainer = Trainer(model, None, None, utils.get_torch_device(), os.path.join(arguments.output_dir))

    frame_indexes, box_predictions, class_predictions = trainer.evaluate(dataloader)

    filtered_detections = filter_detections(torch.Tensor(frame_indexes), torch.Tensor(box_predictions), torch.Tensor(class_predictions)[..., :-1], 0.1)
    labels = load_labels(dataset, filtered_detections)

    # mean_average_precision = mean_average_precision(filtered_detections, dataset, 0.5)

    # output_dict, all_class_labels, all_box_labels = create_task_1_output_format(dataset, dataloader, class_predictions, box_predictions)
    #
    # logging.info("Saving pkl predictions to %s" % os.path.join(arguments.output_dir, EVALUATION_PREDICTION_PKL_FILENAME))
    # utils.write_pkl_file(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_PKL_FILENAME), output_dict)
    #
    # logging.info("Saving json predictions to %s" % os.path.join(arguments.output_dir, EVALUATION_PREDICTION_JSON_FILENAME))
    # utils.write_json_file(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_JSON_FILENAME), output_dict)


def calculate_metrics(dataset, arguments):
    if os.path.isfile(os.path.join(arguments.output_dir, EVALUATION_METRICS_FILENAME)):
        logging.info("Skipping calculation metrics for %s, already exists." % (os.path.join(arguments.output_dir, EVALUATION_METRICS_FILENAME),))
        return

    logging.info("Calculating metrics.")


def main(arguments):
    utils.seed_everything(arguments.seed)

    logger.initLogging(arguments.log_level)
    logging.info("Beginning evaluating task 1.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    dataset = RoadRDataset(VALID_VIDEOS, TRAIN_VALIDATION_DATA_PATH, arguments.image_resize, arguments.num_queries, max_frames=arguments.max_frames)

    evaluate_dataset(dataset, arguments)
    calculate_metrics(dataset, arguments)


def _load_args():
    parser = argparse.ArgumentParser(description="Evaluating Task 1.")

    parser.add_argument("--seed", dest="seed",
                        action="store", type=int, default=4,
                        help="Seed for random number generator.")
    parser.add_argument("--image-resize", dest="image_resize",
                        action="store", type=float, default=1.0,
                        help="Resize factor for all images.")
    parser.add_argument("--num-queries", dest="num_queries",
                        action="store", type=int, default=20,
                        help="Number of object queries, ie detection slot, in a frame.")
    parser.add_argument("--max-frames", dest="max_frames",
                        action="store", type=int, default=0,
                        help="Maximum number of frames to use from each videos. Default is 0, which uses all frames.")
    parser.add_argument("--batch-size", dest="batch_size",
                        action="store", type=int, default=16,
                        help="Batch size.")
    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default="INFO",
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--saved-model-path", dest="saved_model_path",
                        action="store", type=str, default=os.path.join(BASE_RESULTS_DIR, TASK_NAME, TRAINED_MODEL_DIR, TRAINED_MODEL_FILENAME),
                        help="Path to model parameters to load.")
    parser.add_argument("--output-dir", dest="output_dir",
                        action="store", type=str, default=os.path.join(BASE_RESULTS_DIR, TASK_NAME, TRAINED_MODEL_DIR),
                        help="Directory to save results to.")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
