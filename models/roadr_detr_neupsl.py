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
from models.hungarian_match import hungarian_match
from models.losses import binary_cross_entropy_with_logits
from models.losses import pairwise_generalized_box_iou
from models.losses import pairwise_l2_loss
from models.model_utils import save_model_state
from utils import BASE_CLI_DIR
from utils import BASE_RESULTS_DIR
from utils import LABEL_CONFIDENCE_THRESHOLD
from utils import NUM_CLASSES
from utils import NUM_NEUPSL_QUERIES
from utils import NUM_QUERIES
from utils import NUM_SAVED_IMAGES
from utils import SEED
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import TRAINED_MODEL_DIR
from utils import TRAINED_MODEL_FILENAME
from utils import VIDEO_PARTITIONS


TASK_NAME = "task1"

LOAD_PRE_TRAINED_MODEL_PATH = os.path.join(BASE_RESULTS_DIR, TASK_NAME, TRAINED_MODEL_DIR, TRAINED_MODEL_FILENAME)
LOAD_NEUPSL_MODEL_PATH = os.path.join(BASE_RESULTS_DIR, TASK_NAME, TRAINED_MODEL_DIR, "neuspl_evaluation", TRAINED_MODEL_FILENAME)
LOAD_PSL_LABELS_PATH = os.path.join(BASE_CLI_DIR, "inferred-predicates", "LABEL.txt")
OUT_DIR = os.path.join(BASE_RESULTS_DIR, TASK_NAME, TRAINED_MODEL_DIR, "neuspl_evaluation")

LABELED_VIDEOS = VIDEO_PARTITIONS[TASK_NAME]["VALID"]


class RoadRDETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self.application = None

        self.model = None
        self.dataset = None
        self.dataloader = None
        self.optimizer = None
        self.scheduler = None

        self.batch_predictions = None
        self.batch_predictions_formatted = None

        self.iterator = None
        self.current_batch = None

        self.all_box_predictions = []
        self.all_class_predictions = []
        self.all_frame_indexes = []

        utils.seed_everything(SEED)
        logger.initLogging(logging_level=logging.INFO)

    def internal_init(self, application, options={}):
        logging.info("Initializing neural model for application: {0}".format(application))
        self.application = application

        self.model = task_1_model()

        model_path = LOAD_PRE_TRAINED_MODEL_PATH

        if self.application == "learning":
            self.model.load_state_dict(torch.load(model_path))
            self.dataset = RoadRDataset(VIDEO_PARTITIONS[TASK_NAME]["TRAIN"], TRAIN_VALIDATION_DATA_PATH, float(options["image-resize"]), max_frames=int(options["max-frames"]))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(options["learning-rate"]), weight_decay=float(options["weight-decay"]))
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(options["step-size"]), gamma=float(options["gamma"]))
        else:
            if os.path.isfile(LOAD_NEUPSL_MODEL_PATH):
                logging.info("Saved NeuPSL model found, loading: {0}".format(LOAD_NEUPSL_MODEL_PATH))
                model_path = LOAD_NEUPSL_MODEL_PATH

            self.model.load_state_dict(torch.load(model_path))
            self.dataset = RoadRDataset(VIDEO_PARTITIONS[TASK_NAME]["VALID"], TRAIN_VALIDATION_DATA_PATH, float(options["image-resize"]), max_frames=int(options["max-frames"]))
            self.optimizer = None
            self.scheduler = None

        self.dataloader = DataLoader(self.dataset, batch_size=int(options["batch-size"]), shuffle=False)

        return {}

    def internal_fit(self, data, gradients, options={}):
        self.set_model_application(True)

        batch = [b.to(utils.get_torch_device()) for b in self.current_batch]

        structured_gradients = float(options["alpha"]) * self.format_batch_gradients(gradients, len(self.current_batch[1]))

        self.optimizer.zero_grad(set_to_none=True)
        self.batch_predictions["logits"].backward(structured_gradients, retain_graph=True)

        loss, results = self._compute_loss(batch)
        loss.backward(retain_graph=True)

        self.post_gradient_computation()
        self.optimizer.step()

        return results

    def internal_predict(self, data, options={}):
        self.set_model_application(options["learn"])

        frame_ids, pixel_values, pixel_mask, labels, boxes = [b.to(utils.get_torch_device()) for b in self.current_batch]

        self.batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})
        if options["learn"] == "learn":
            self.all_frame_indexes.extend(frame_ids.cpu().tolist())

        return self.format_batch_predictions(self.batch_predictions["logits"].detach().cpu(), self.batch_predictions["pred_boxes"].detach().cpu(), options=options), {}

    def internal_eval(self, data, options={}):
        if self.application == "learning":
            return {}

        self.set_model_application(False)

        self.format_batch_results(options=options)

        if self.current_batch is None:
            os.makedirs(OUT_DIR, exist_ok=True)

            create_task_1_output_format(self.dataset, self.all_frame_indexes, self.all_class_predictions, self.all_box_predictions, output_dir=OUT_DIR, from_logits=False)

            calculate_metrics(self.dataset, OUT_DIR)

            save_images_with_bounding_boxes(self.dataset, OUT_DIR, True, NUM_SAVED_IMAGES, LABEL_CONFIDENCE_THRESHOLD)

        return {}

    def internal_epoch_start(self, options={}):
        self.iterator = iter(self.dataloader)
        self.next_batch()

    def internal_epoch_end(self, options={}):
        if self.scheduler is not None:
            self.scheduler.step()
            self.save()

    def internal_next_batch(self, options={}):
        self.current_batch = next(self.iterator, None)

    def internal_is_epoch_complete(self, options={}):
        if self.current_batch is None:
            return {"is_epoch_complete": True}
        return {"is_epoch_complete": False}

    def internal_save(self, options={}):
        os.makedirs(OUT_DIR, exist_ok=True)
        save_model_state(self.model, OUT_DIR, TRAINED_MODEL_FILENAME)

    def set_model_application(self, learning):
        if learning:
            self.model.train()
        else:
            self.model.eval()

    def format_batch_gradients(self, gradients, batch_size, options={}):
        formatted_gradients = torch.zeros(size=(batch_size, NUM_QUERIES, NUM_CLASSES + 1), dtype=torch.float32)

        batch_index = -1
        for gradient_index in range(len(gradients)):
            if gradient_index % NUM_NEUPSL_QUERIES == 0:
                batch_index += 1
                if batch_index >= batch_size:
                    break

            query_index = gradient_index % NUM_NEUPSL_QUERIES
            for class_index in range(len(gradients[gradient_index][:-5])):
                formatted_gradients[batch_index][query_index][class_index] = float(gradients[gradient_index][class_index])

        return formatted_gradients.to(utils.get_torch_device())

    def format_batch_predictions(self, logits, boxes, options={}):
        self.batch_predictions_formatted = torch.zeros(size=(int(options["batch-size"]), NUM_NEUPSL_QUERIES, int(options["class-size"])), dtype=torch.float32)

        batch_predictions_sorted_indexes = torch.argsort(logits[:, :, -1], descending=True)
        batch_predictions = torch.cat((torch.sigmoid(logits), boxes), dim=-1)

        for batch_index in range(len(batch_predictions)):
            for box_index in range(NUM_NEUPSL_QUERIES):
                self.batch_predictions_formatted[batch_index][box_index] = batch_predictions[batch_index][batch_predictions_sorted_indexes[batch_index][box_index]]

        return self.batch_predictions_formatted.flatten(start_dim=0, end_dim=1).cpu().detach().numpy().tolist()

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
                    box_predictions[batch_index][query_index][class_index - (int(options["class-size"]) - 4)] = self.batch_predictions_formatted[batch_index][query_index][class_index]
                elif class_index == int(options["class-size"]) - 5:
                    class_predictions[batch_index][query_index][class_index] = self.batch_predictions_formatted[batch_index][query_index][class_index]
                else:
                    class_predictions[batch_index][query_index][class_index] = float(line.strip().split("\t")[-1])

                line_count += 1
                predictions_index += 1

        self.all_box_predictions.extend(box_predictions.tolist())
        self.all_class_predictions.extend(class_predictions.tolist())

    def _compute_loss(self, data, bce_weight: int = 1, giou_weight: int = 2, l2_weight: int = 1):
        frame_ids, pixel_values, pixel_mask, labels, boxes = data

        # Compute the training loss.
        # For the training loss, we need to first compute the matching between the predictions and the ground truth.
        matching = hungarian_match(self.batch_predictions["pred_boxes"], boxes)

        # Compute the classification loss using the matching.
        bce_loss = binary_cross_entropy_with_logits(self.batch_predictions["logits"], labels, matching)

        # Compute the bounding box loss using the matching.
        giou_loss = pairwise_generalized_box_iou(self.batch_predictions["pred_boxes"], boxes, matching)

        # Compute the bounding box l2 loss using the matching.
        l2_loss = pairwise_l2_loss(self.batch_predictions["pred_boxes"], boxes, matching)

        results = {
            "bce_loss": bce_loss.item(),
            "giou_loss": giou_loss.item(),
            "l2_loss": l2_loss.item(),
            "loss": (bce_weight * bce_loss + giou_weight * giou_loss + l2_weight * l2_loss).item()
        }

        return bce_weight * bce_loss + giou_weight * giou_loss + l2_weight * l2_loss, results

    def post_gradient_computation(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)