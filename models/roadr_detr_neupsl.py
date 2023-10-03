#!/usr/bin/env python3
import logging
import os
import sys

import numpy
import pslpython.deeppsl.model
import torch.nn

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger
import utils

from data.roadr_dataset import RoadRDataset
from experiments.task1_evaluate import create_task_1_output_format
from experiments.task1_evaluate import calculate_metrics
from experiments.task1_pretrain import task_1_model
from models.analysis import save_images_with_bounding_boxes
from utils import BASE_CLI_DIR
from utils import BASE_RESULTS_DIR
from utils import LABEL_CONFIDENCE_THRESHOLD
from utils import NUM_NEUPSL_QUERIES
from utils import NUM_QUERIES
from utils import NUM_SAVED_IMAGES
from utils import SEED
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import TRAINED_MODEL_DIR
from utils import TRAINED_MODEL_FILENAME
from utils import VIDEO_PARTITIONS


TASK_NAME = "task1"

LOAD_MODEL_PATH = os.path.join(BASE_RESULTS_DIR, TASK_NAME, TRAINED_MODEL_DIR, TRAINED_MODEL_FILENAME)
LOAD_PSL_LABELS_PATH = os.path.join(BASE_CLI_DIR, "inferred-predicates", "LABEL.txt")
OUT_DIR = os.path.join(BASE_RESULTS_DIR, TASK_NAME, TRAINED_MODEL_DIR, "neuspl_evaluation")

LABELED_VIDEOS = VIDEO_PARTITIONS[TASK_NAME]["VALID"]

BATCH_INCOMPLETE = 0
BATCH_COMPLETE = 1


class RoadRDETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self.application = None

        self.dataset = None
        self.dataloader = None
        self.iterator = None
        self.model = None

        self.batch_predictions = None

        self.epoch_complete = False

        self.all_box_predictions = []
        self.all_class_predictions = []
        self.all_frame_indexes = []

        utils.seed_everything(SEED)
        logger.initLogging(logging_level=logging.INFO)

    def internal_init_model(self, application, options={}):
        logging.info("Initializing neural model for application: {0}".format(application))
        self.application = application

        self.dataset = RoadRDataset(LABELED_VIDEOS, TRAIN_VALIDATION_DATA_PATH, float(options['image-resize']), max_frames=int(options['max-frames']))
        self.dataloader = DataLoader(self.dataset, batch_size=int(options["batch-size"]), shuffle=False)
        self.iterator = iter(self.dataloader)
        self.current_batch = next(self.iterator, None)

        self.model = task_1_model()
        self.model.load_state_dict(torch.load(LOAD_MODEL_PATH))

        self.batch_predictions = torch.empty(size=(int(options["batch-size"]), NUM_NEUPSL_QUERIES, int(options["class-size"])), dtype=torch.float32)

        return {}

    def internal_fit(self, data, gradients, options={}):
        self.optimizer.zero_grad(set_to_none=True)

        loss = self._compute_loss()

        total_loss = self.bce_weight * loss["bce_loss"] + self.giou_weight * loss["giou_loss"]
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return {"total_loss": total_loss.item(),
                "bce_loss": loss["bce_loss"].item(),
                "giou_loss": loss["giou_loss"].item()}

    def internal_predict(self, data, options={}):
        self.set_model_application()

        frame_ids, pixel_values, pixel_mask, labels, boxes = [b.to(utils.get_torch_device()) for b in self.current_batch]

        batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})
        self.all_frame_indexes.extend(frame_ids.cpu().tolist())

        return self.format_batch_predictions(batch_predictions["logits"].detach().cpu(), batch_predictions["pred_boxes"].detach().cpu(), options=options), {}

    def internal_eval(self, data, options={}):
        self.set_model_application()

        self.format_batch_results(options=options)

        if self.epoch_complete:
            os.makedirs(OUT_DIR, exist_ok=True)

            create_task_1_output_format(self.dataset, self.all_frame_indexes, self.all_class_predictions, self.all_box_predictions, output_dir=OUT_DIR, from_logits=False)

            calculate_metrics(self.dataset, OUT_DIR)

            save_images_with_bounding_boxes(self.dataset, OUT_DIR, True, NUM_SAVED_IMAGES, LABEL_CONFIDENCE_THRESHOLD)

        return {}

    def internal_next_batch(self, options={}):
        self.current_batch = next(self.iterator, None)
        if self.current_batch is not None:
            return

        self.epoch_complete = True
        self.iterator = iter(self.dataloader)
        self.current_batch = next(self.iterator, None)


    def internal_is_epoch_complete(self, options={}):
        if self.epoch_complete:
            self.epoch_complete = False
            return {"is_epoch_complete": True}
        return {"is_epoch_complete": False}

    def internal_save(self, options={}):
        return {}

    def set_model_application(self):
        if self.application == "inference":
            self.model.eval()
        else:
            self.model.train()

    def format_batch_predictions(self, logits, boxes, options={}):
        self.batch_predictions = torch.zeros(size=(int(options["batch-size"]), NUM_NEUPSL_QUERIES, int(options["class-size"])), dtype=torch.float32)

        batch_predictions_sorted_indexes = torch.argsort(logits[:, :, -1], descending=True)
        batch_predictions = torch.cat((torch.sigmoid(logits), boxes), dim=-1)

        for batch_index in range(len(batch_predictions)):
            for box_index in range(NUM_NEUPSL_QUERIES):
                self.batch_predictions[batch_index][box_index] = batch_predictions[batch_index][batch_predictions_sorted_indexes[batch_index][box_index]]

        return self.batch_predictions.flatten(start_dim=0, end_dim=1).cpu().detach().numpy().tolist()

    def format_batch_results(self, options={}):
        box_predictions = numpy.zeros(shape=(int(options["batch-size"]), NUM_QUERIES, 4), dtype=numpy.float32)
        class_predictions = numpy.zeros(shape=(int(options["batch-size"]), NUM_QUERIES, int(options["class-size"]) - 4), dtype=numpy.float32)

        line_count = 0
        predictions_index = 0
        with open(LOAD_PSL_LABELS_PATH, "r") as file:
            for line in file.readlines():
                if line_count % (int(options["class-size"]) * NUM_NEUPSL_QUERIES) == 0 and predictions_index > 0:
                    predictions_index += int(options["class-size"]) * (NUM_QUERIES - NUM_NEUPSL_QUERIES)

                batch_index = predictions_index // (int(options["class-size"]) * NUM_QUERIES)
                query_index = predictions_index % (int(options["class-size"]) * NUM_QUERIES) // int(options["class-size"])
                class_index = predictions_index % int(options["class-size"])

                if class_index > int(options["class-size"]) - 5:
                    box_predictions[batch_index][query_index][class_index - (int(options["class-size"]) - 4)] = self.batch_predictions[batch_index][query_index][class_index]
                elif class_index == int(options["class-size"]) - 5:
                    class_predictions[batch_index][query_index][class_index] = self.batch_predictions[batch_index][query_index][class_index]
                else:
                    class_predictions[batch_index][query_index][class_index] = float(line.strip().split("\t")[-1])

                line_count += 1
                predictions_index += 1

        self.all_box_predictions.extend(box_predictions.tolist())
        self.all_class_predictions.extend(class_predictions.tolist())

    def _compute_loss(self):
        results = {}

        frames, train_images, labels, boxes = self.current_batch

        # Compute the matching between the predictions and the ground truth.
        matching = hungarian_match(self.predictions["boxes"], boxes)

        # Compute the classification loss using the matching.
        results["bce_loss"] = binary_cross_entropy_with_logits(self.predictions["class_probabilities"], labels, matching)

        # Compute the bounding box loss using the matching.
        results["giou_loss"] = pairwise_generalized_box_iou(self.predictions["boxes"], boxes, matching)

        return results
