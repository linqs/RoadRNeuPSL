#!/usr/bin/env python3
import logging
import os
import sys

import pslpython.deeppsl.model
import torch.nn

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logger
import utils

from data.roadr_dataset import RoadRDataset
from experiments.task1_evaluate import create_task_1_output_format
from experiments.task1_evaluate import format_saved_predictions
from experiments.task1_evaluate import sort_by_confidence
from experiments.task1_pretrain import task_1_model
from models.analysis import save_images_with_bounding_boxes
from models.evaluation import filter_detections
from models.evaluation import load_ground_truth_for_detections
from models.evaluation import mean_average_precision
from utils import EVALUATION_METRICS_FILENAME
from utils import EVALUATION_PREDICTION_JSON_FILENAME
from utils import EVALUATION_PREDICTION_PKL_FILENAME

logger.initLogging(logging_level=logging.DEBUG)

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "..", "data")
DATA_FILE_NAME = 'road_trainval_v1.0.json'

LABELED_VIDEOS = ["2014-06-26-09-53-12_stereo_centre_02",
                  "2014-11-25-09-18-32_stereo_centre_04",
                  "2015-02-13-09-16-26_stereo_centre_02"]

MAX_FRAMES = 10
BATCH_SIZE = 4
NUM_CLASSES = 41
NUM_BOXES = 4
NUM_QUERIES = 100
NUM_NEUPSL_QUERIES = 20
IMAGE_RESIZE = 0.4

IOU_THRESHOLD = 0.5
LABEL_CONFIDENCE_THRESHOLD = 0.025

SAVE_MODEL_PATH = os.path.join(THIS_DIR, "..", "results", "task1", "trained_model", "trained_model_parameters_checkpoint.pt")
OUT_DIR = os.path.join(THIS_DIR, "..", "results", "task1", "trained_model", "neuspl_evaluation")


class RoadRDETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    """
    DETR Model from: <https://arxiv.org/pdf/2005.12872.pdf>
    """
    def __init__(self):
        super().__init__()
        self.application = None

        self.dataset = None
        self.dataloader = None
        self.iterator = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        self.current_batch = None
        self.batch_predictions = None

        self.batch_index = 0

        self.all_box_predictions = []
        self.all_logits = []
        self.all_frame_indexes = []

    def internal_init_model(self, application, options={}):
        logging.info("Initializing neural model for application: {0}".format(application))
        self.application = application

        self.dataset = RoadRDataset(LABELED_VIDEOS, os.path.join(DATA_DIR, DATA_FILE_NAME), IMAGE_RESIZE, NUM_QUERIES, max_frames=MAX_FRAMES)
        self.dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=False)
        self.iterator = iter(self.dataloader)

        self.model = task_1_model()
        self.model.load_state_dict(torch.load(SAVE_MODEL_PATH))

        return {}

    def internal_fit(self, data, gradients, options={}):
        return {}

    def internal_predict(self, data, options={}):
        self.set_model_application()

        self.current_batch = [b.to(utils.get_torch_device()) for b in next(self.iterator)]

        frame_ids, pixel_values, pixel_mask, labels, boxes = self.current_batch

        self.batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})
        self.all_frame_indexes.extend(frame_ids.cpu().tolist())
        self.filter_batch_predictions()

        return self.format_predictions(), {}

    def internal_eval(self, data, predictions, options={}):
        self.set_model_application()

        for batch_index in range(BATCH_SIZE):
            self.all_box_predictions.append([])
            self.all_logits.append([])
            for query_index in range(NUM_QUERIES):
                if query_index < NUM_NEUPSL_QUERIES:
                    self.all_box_predictions[-1].append(predictions[batch_index * NUM_NEUPSL_QUERIES + query_index][-4:].tolist())
                    self.all_logits[-1].append(predictions[batch_index * NUM_NEUPSL_QUERIES + query_index][:-4].tolist())
                else:
                    self.all_box_predictions[-1].append([0] * NUM_BOXES)
                    self.all_logits[-1].append([0] * (NUM_CLASSES + 1))

        if self.batch_index == 5:
            output_dict = create_task_1_output_format(self.dataset, self.all_frame_indexes, self.all_logits, self.all_box_predictions, from_logits=False)

            os.makedirs(OUT_DIR, exist_ok=True)

            logging.info("Saving pkl predictions to %s" % os.path.join(OUT_DIR, EVALUATION_PREDICTION_PKL_FILENAME))
            utils.write_pkl_file(os.path.join(OUT_DIR, EVALUATION_PREDICTION_PKL_FILENAME), output_dict)

            logging.info("Saving json predictions to %s" % os.path.join(OUT_DIR, EVALUATION_PREDICTION_JSON_FILENAME))
            utils.write_json_file(os.path.join(OUT_DIR, EVALUATION_PREDICTION_JSON_FILENAME), output_dict)

            logging.info("Loading predictions.")
            loaded_predictions = utils.load_json_file(os.path.join(OUT_DIR, EVALUATION_PREDICTION_JSON_FILENAME))
            frame_indexes, class_predictions, box_predictions = format_saved_predictions(loaded_predictions, self.dataset)

            logging.info("Calculating metrics.")
            filtered_detections, filtered_detection_indexes = filter_detections(torch.Tensor(frame_indexes), torch.Tensor(box_predictions), torch.Tensor(class_predictions), IOU_THRESHOLD, LABEL_CONFIDENCE_THRESHOLD)
            filtered_detections_ground_truth = load_ground_truth_for_detections(self.dataset, filtered_detection_indexes)

            mean_avg_prec = mean_average_precision(filtered_detections_ground_truth, filtered_detections, IOU_THRESHOLD)
            logging.info("Mean average precision: %s" % mean_avg_prec)

            logging.info("Saving metrics to %s" % os.path.join(OUT_DIR, EVALUATION_METRICS_FILENAME))
            utils.write_json_file(os.path.join(OUT_DIR, EVALUATION_METRICS_FILENAME), {"mean_avg_prec": mean_avg_prec})

            save_images_with_bounding_boxes(self.dataset, OUT_DIR, True, 3, LABEL_CONFIDENCE_THRESHOLD)

        self.batch_index += 1

        return {}

    def internal_save(self, options={}):
        return {}

    def set_model_application(self):
        if self.application == "inference":
            self.model.eval()
        else:
            self.model.train()

    def format_predictions(self):
        self.batch_predictions["logits"] = torch.sigmoid(self.batch_predictions["logits"])
        batch_predictions = torch.cat((self.batch_predictions["logits"], self.batch_predictions["pred_boxes"]), dim=-1)
        batch_predictions = batch_predictions.flatten(start_dim=0, end_dim=1)
        return batch_predictions.cpu().detach().numpy().tolist()

    def filter_batch_predictions(self):
        batch_filterd_logits = []
        batch_filtered_boxes = []
        for logit, box in zip(self.batch_predictions['logits'].cpu().tolist(), self.batch_predictions['pred_boxes'].cpu().tolist()):
            filtered_logits, filtered_boxes = sort_by_confidence(logit, box)
            batch_filterd_logits.append(filtered_logits[:NUM_NEUPSL_QUERIES])
            batch_filtered_boxes.append(filtered_boxes[:NUM_NEUPSL_QUERIES])

        self.batch_predictions['logits'] = torch.tensor(batch_filterd_logits).to(utils.get_torch_device())
        self.batch_predictions['pred_boxes'] = torch.tensor(batch_filtered_boxes).to(utils.get_torch_device())
