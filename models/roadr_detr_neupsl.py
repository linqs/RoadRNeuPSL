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
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.batch_predictions = None

        self.iterator = None
        self.current_batch = None

        self.all_box_predictions = []
        self.all_class_predictions = []
        self.all_frame_indexes = []

        utils.seed_everything(SEED)
        logger.initLogging(logging_level=logging.INFO)

    def internal_init_model(self, application, options={}):
        logging.info("Initializing neural model for application: {0}".format(application))
        self.model = task_1_model()
        self.model.load_state_dict(torch.load(LOAD_MODEL_PATH))

        self.dataset = RoadRDataset(LABELED_VIDEOS, TRAIN_VALIDATION_DATA_PATH, float(options['image-resize']), max_frames=int(options['max-frames']))
        self.dataloader = DataLoader(self.dataset, batch_size=int(options["batch-size"]), shuffle=False)

        self.batch_predictions = torch.empty(size=(int(options["batch-size"]), NUM_NEUPSL_QUERIES, int(options["class-size"])), dtype=torch.float32)

        return {}

    def internal_fit(self, data, gradients, options={}):
        self.set_model_application(options['learn'])

        structured_gradients = float(options["alpha"]) * torch.tensor(gradients.astype(numpy.float32), dtype=torch.float32, device=utils.get_torch_device())

        batch = [b.to(utils.get_torch_device()) for b in self.current_batch]

        self.optimizer.zero_grad(set_to_none=True)

        loss = self._compute_loss(batch)

        loss.backward()
        self.post_gradient_computation()
        epoch_loss += loss.item()

        return results

    self._prepare_data(data, options=options)



    dino_loss = (1 - float(options["alpha"])) * (self._dino_loss(self._teacher_predictions_1, self._student_predictions_1)
                                                      + self._dino_loss(self._teacher_predictions_2, self._student_predictions_2)) / 2.0

    self._optimizer.zero_grad()

    dino_loss.backward(retain_graph=True)
    self._student_predictions_1.backward(structured_gradients, retain_graph=True)

    torch.nn.utils.clip_grad_norm_(self._student_model.parameters(), 3.0)

    self._optimizer.step()

    # EMA updates for the teacher
    with torch.no_grad():
        for param_student, param_teacher in zip(self._student_model.parameters(), self._teacher_model.parameters()):
            param_teacher.data.mul_(self._teacher_parameter_momentum).add_((1 - self._teacher_parameter_momentum) * param_student.detach().data)

        self._teacher_center = (self._teacher_center_momentum * self._teacher_center
                                + (1.0 - self._teacher_center_momentum) * torch.cat([self._teacher_predictions_1, self._teacher_predictions_2], dim=0).mean(dim=0))

    # Compute the new training loss.
    new_output = self._student_model(self._features)

    loss = torch.nn.functional.cross_entropy(new_output, self._digit_labels).item()

    results = {"training_classification_loss": loss,
               "dino_loss": dino_loss.item(),
               "teacher_center": self._teacher_center.cpu().detach().numpy().tolist(),
               "struct gradient_norm_2": torch.norm(structured_gradients, 2).item(),
               "struct gradient_norm_infty": torch.norm(structured_gradients, torch.inf).item()}

    return results

    def internal_predict(self, data, options={}):
        self.set_model_application(options['learn'])

        frame_ids, pixel_values, pixel_mask, labels, boxes = [b.to(utils.get_torch_device()) for b in self.current_batch]

        batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})
        self.all_frame_indexes.extend(frame_ids.cpu().tolist())

        return self.format_batch_predictions(batch_predictions["logits"].detach().cpu(), batch_predictions["pred_boxes"].detach().cpu(), options=options), {}

    def internal_eval(self, data, options={}):
        self.set_model_application(options['learn'])

        self.format_batch_results(options=options)

        if self.current_batch is None:
            os.makedirs(OUT_DIR, exist_ok=True)

            create_task_1_output_format(self.dataset, self.all_frame_indexes, self.all_class_predictions, self.all_box_predictions, output_dir=OUT_DIR, from_logits=False)

            calculate_metrics(self.dataset, OUT_DIR)

            save_images_with_bounding_boxes(self.dataset, OUT_DIR, True, NUM_SAVED_IMAGES, LABEL_CONFIDENCE_THRESHOLD)

        return {}

    def internal_epoch_start(self):
        self.iterator = iter(self.dataloader)
        self.next_batch()

    def internal_epoch_end(self):
        self.scheduler.step()

    def internal_next_batch(self, options={}):
        self.current_batch = next(self.iterator, None)

    def internal_is_epoch_complete(self, options={}):
        if self.current_batch is None:
            return {"is_epoch_complete": True}
        return {"is_epoch_complete": False}

    def internal_save(self, options={}):
        return {}

    def set_model_application(self, application):
        if application == "learn":
            self.model.train()
        else:
            self.model.eval()

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

    def _compute_loss(self, data, bce_weight: int = 1, giou_weight: int = 2, l2_weight: int = 1):
        frame_ids, pixel_values, pixel_mask, labels, boxes = data

        # Compute the predictions for the provided batch.
        self.batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})

        # Compute the training loss.
        # For the training loss, we need to first compute the matching between the predictions and the ground truth.
        matching = hungarian_match(self.batch_predictions["pred_boxes"], boxes)

        # Compute the classification loss using the matching.
        bce_loss = binary_cross_entropy_with_logits(self.batch_predictions["logits"], labels, matching)

        # Compute the bounding box loss using the matching.
        giou_loss = pairwise_generalized_box_iou(self.batch_predictions["pred_boxes"], boxes, matching)

        # Compute the bounding box l2 loss using the matching.
        l2_loss = pairwise_l2_loss(self.batch_predictions["pred_boxes"], boxes, matching)

        return bce_weight * bce_loss + giou_weight * giou_loss + l2_weight * l2_loss