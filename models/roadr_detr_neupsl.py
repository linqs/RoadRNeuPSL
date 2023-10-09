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

from data.stream_roadr_dataset import RoadRDataset
from experiments.evaluate import save_logits_and_labels
from experiments.evaluate import calculate_metrics
from experiments.pretrain import build_model
from models.analysis import save_images_with_bounding_boxes
from models.hungarian_match import hungarian_match
from models.losses import binary_cross_entropy_with_logits
from models.losses import pairwise_generalized_box_iou
from models.losses import pairwise_l1_loss
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
from utils import NEURAL_TRAINED_MODEL_DIR
from utils import NEURAL_TRAINED_MODEL_FILENAME
from utils import NEUPSL_TRAINED_MODEL_DIR
from utils import NEUPSL_TRAINED_MODEL_FILENAME
from utils import VIDEO_PARTITIONS


LOAD_PSL_LABELS_PATH = os.path.join(BASE_CLI_DIR, "inferred-predicates", "LABEL.txt")


class RoadRDETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self.application = None

        self.out_dir = None

        self.model = None
        self.dataset = None
        self.dataloader = None
        self.optimizer = None
        self.scheduler = None

        self.batch_predictions = None
        self.batch_predictions_formatted = None

        self.iterator = None
        self.current_batch = None
        self.gpu_batch = None

        self.all_box_predictions = []
        self.all_class_predictions = []
        self.all_frame_indexes = []

        utils.seed_everything(SEED)
        logger.initLogging(logging_level=logging.INFO)

    def internal_init(self, application, options={}):
        logging.info("Initializing neural model for application: {0}".format(application))
        self.application = application

        self.out_dir = os.path.join(BASE_RESULTS_DIR, options["task-name"], NEUPSL_TRAINED_MODEL_DIR, "neupsl_evaluation")

        self.model = build_model()

        neural_trained_model_path = os.path.join(BASE_RESULTS_DIR, options["task-name"],
                                                 NEURAL_TRAINED_MODEL_DIR, NEURAL_TRAINED_MODEL_FILENAME)

        if self.application == "learning":
            self.model.load_state_dict(torch.load(neural_trained_model_path))
            self.dataset = RoadRDataset(VIDEO_PARTITIONS[options["task-name"]]["TRAIN"], TRAIN_VALIDATION_DATA_PATH,
                                        float(options["image-resize"]),
                                        max_frames=int(options["max-frames"]))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(options["learning-rate"]), weight_decay=float(options["weight-decay"]))
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(options["step-size"]), gamma=float(options["gamma"]))
        else:
            neupsl_trained_model_path = os.path.join(self.out_dir, NEUPSL_TRAINED_MODEL_FILENAME)
            if os.path.isfile(neupsl_trained_model_path):
                logging.info("Saved NeuPSL model found, loading: {0}".format(neupsl_trained_model_path))
                trained_model_path = neupsl_trained_model_path
            else:
                trained_model_path = neural_trained_model_path

            self.model.load_state_dict(torch.load(trained_model_path, map_location=utils.get_torch_device()))
            self.dataset = RoadRDataset(VIDEO_PARTITIONS[options["task-name"]][options["inference_split"]], TRAIN_VALIDATION_DATA_PATH, float(options["image-resize"]),
                                        max_frames=int(options["max-frames"]), test=(options["inference_split"] == "TEST"))
            self.optimizer = None
            self.scheduler = None

        self.dataloader = DataLoader(self.dataset, batch_size=int(options["batch-size"]),
                                     shuffle=False, num_workers=int(os.cpu_count()) - 2,
                                     prefetch_factor=4, persistent_workers=True)

        return {}

    def internal_fit(self, data, gradients, options={}):
        self.set_model_application(True)

        structured_gradients = float(options["alpha"]) * self.format_batch_gradients(gradients, len(self.current_batch[1]))

        self.optimizer.zero_grad(set_to_none=True)
        self.batch_predictions["logits"].backward(structured_gradients, retain_graph=True)

        loss, results = self._compute_loss(self.gpu_batch)
        loss = (1 - float(options["alpha"])) * loss
        loss.backward(retain_graph=True)

        self.post_gradient_computation()
        self.optimizer.step()

        return results

    def internal_predict(self, data, options={}):
        self.set_model_application(options["learn"])

        frame_ids, pixel_values, pixel_mask, labels, boxes = self.gpu_batch

        self.batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})
        if not options["learn"]:
            self.all_frame_indexes.extend(frame_ids.cpu().tolist())

        return self.format_batch_predictions(self.batch_predictions["logits"].detach().cpu(), self.batch_predictions["pred_boxes"].detach().cpu(), options=options), {}

    def internal_eval(self, data, options={}):
        if self.application == "learning":
            return {'loss': 0.0}

        self.set_model_application(False)

        self.format_batch_results(options=options)

        return {'loss': 0.0}

    def internal_epoch_start(self, options={}):
        self.iterator = iter(self.dataloader)
        self.next_batch()

    def internal_epoch_end(self, options={}):
        if self.scheduler is not None:
            # Learning
            self.scheduler.step()
            self.save()
        else:
            # Inference
            os.makedirs(self.out_dir, exist_ok=True)
            save_logits_and_labels(self.dataset, self.all_frame_indexes, self.all_class_predictions, self.all_box_predictions,
                                   output_dir=self.out_dir, from_logits=False)

            if options["inference_split"] == "VALID":
                calculate_metrics(self.dataset, self.out_dir)

            save_images_with_bounding_boxes(self.dataset, self.out_dir, True,
                                            NUM_SAVED_IMAGES, LABEL_CONFIDENCE_THRESHOLD,
                                            write_ground_truth=(options["inference_split"] == "VALID"),
                                            test=(options["inference_split"] == "TEST"))

    def internal_next_batch(self, options={}):
        self.current_batch = next(self.iterator, None)
        if self.current_batch is not None:
            self.gpu_batch = [b.to(utils.get_torch_device()) for b in self.current_batch]

    def internal_is_epoch_complete(self, options={}):
        if self.current_batch is None:
            return {"is_epoch_complete": True}
        return {"is_epoch_complete": False}

    def internal_save(self, options={}):
        os.makedirs(self.out_dir, exist_ok=True)
        save_model_state(self.model, self.out_dir, NEUPSL_TRAINED_MODEL_FILENAME)

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

    def _compute_loss(self, data, bce_weight: int = 1, giou_weight: int = 5, l1_weight: int = 2):
        frame_ids, pixel_values, pixel_mask, labels, boxes = data

        # Compute the training loss.
        # For the training loss, we need to first compute the matching between the predictions and the ground truth.
        matching = hungarian_match(self.batch_predictions["pred_boxes"], boxes)

        # Compute the classification loss using the matching.
        bce_loss = binary_cross_entropy_with_logits(self.batch_predictions["logits"], labels, matching)

        # Compute the bounding box loss using the matching.
        giou_loss = pairwise_generalized_box_iou(self.batch_predictions["pred_boxes"], boxes, matching)

        # Compute the bounding box l2 loss using the matching.
        l1_loss = pairwise_l1_loss(self.batch_predictions["pred_boxes"], boxes, matching)

        results = {
            "bce_loss": bce_loss.item(),
            "giou_loss": giou_loss.item(),
            "l1_loss": l1_loss.item(),
            "loss": (bce_weight * bce_loss + giou_weight * giou_loss + l1_weight * l1_loss).item()
        }

        return bce_weight * bce_loss + giou_weight * giou_loss + l1_weight * l1_loss, results

    def post_gradient_computation(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)