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

from data.roadr_dataset import RoadRDataset
from experiments.evaluate import calculate_metrics
from experiments.evaluate import save_logits_and_labels
from experiments.pretrain import build_model
from models.analysis import save_images_with_bounding_boxes
from models.losses import detr_loss
from utils import get_torch_device
from utils import load_json_file
from utils import save_model_state
from utils import seed_everything
from utils import BASE_CLI_DIR
from utils import BASE_RESULTS_DIR
from utils import BOX_CONFIDENCE_THRESHOLD
from utils import LABEL_CONFIDENCE_THRESHOLD
from utils import NEUPSL_TEST_INFERENCE_DIR
from utils import NEUPSL_TRAINED_MODEL_DIR
from utils import NEUPSL_TRAINED_MODEL_FILENAME
from utils import NEUPSL_VALID_INFERENCE_DIR
from utils import NEURAL_TEST_INFERENCE_DIR
from utils import NEURAL_TRAINED_MODEL_DIR
from utils import NEURAL_TRAINED_MODEL_FILENAME
from utils import NEURAL_VALID_INFERENCE_DIR
from utils import NUM_CLASSES
from utils import NUM_NEUPSL_QUERIES
from utils import NUM_QUERIES
from utils import NUM_SAVED_IMAGES
from utils import PREDICTION_PROBABILITIES_WITH_CONFIDENCE_JSON_FILENAME
from utils import SEED
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import VIDEO_PARTITIONS


LOAD_PSL_LABELS_PATH = os.path.join(BASE_CLI_DIR, "inferred-predicates", "LABEL.txt")


class RoadRDETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self.application = None

        self.evaluation_out_dir = None
        self.model_out_dir = None

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

        seed_everything(SEED)
        logger.initLogging(logging_level=logging.INFO)

    def internal_init(self, application, options={}):
        logging.info("Initializing neural model for application: {0}".format(application))
        self.application = application

        self.model_out_dir = os.path.join(BASE_RESULTS_DIR, options["task-name"], NEUPSL_TRAINED_MODEL_DIR)

        if options["load-predictions"] == "true":
            assert self.application != "learning"
            if (options["inference_split"] == "VALID"):
                predictions_path = os.path.join(BASE_RESULTS_DIR, options["task-name"], NEURAL_VALID_INFERENCE_DIR, "evaluation", PREDICTION_PROBABILITIES_WITH_CONFIDENCE_JSON_FILENAME)
            elif (options["inference_split"] == "TEST"):
                predictions_path = os.path.join(BASE_RESULTS_DIR, options["task-name"], NEURAL_TEST_INFERENCE_DIR, "evaluation", PREDICTION_PROBABILITIES_WITH_CONFIDENCE_JSON_FILENAME)
            else:
                raise ValueError("Invalid inference split: {0}".format(options["inference_split"]))
            self.model = LoadPredictionsModel(predictions_path)
        else:
            self.model = build_model()

        neural_trained_model_path = os.path.join(BASE_RESULTS_DIR, options["task-name"],
                                                 NEURAL_TRAINED_MODEL_DIR, NEURAL_TRAINED_MODEL_FILENAME)

        if self.application == "learning":
            if os.path.isfile(neural_trained_model_path):
                self.model.load_state_dict(torch.load(neural_trained_model_path))
            self.dataset = RoadRDataset(VIDEO_PARTITIONS[options["task-name"]]["TRAIN"], TRAIN_VALIDATION_DATA_PATH,
                                        float(options["image-resize"]),
                                        max_frames=int(options["max-frames"]))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(options["learning-rate"]), weight_decay=float(options["weight-decay"]))
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(options["step-size"]), gamma=float(options["gamma"]))
        else:
            use_neural_trained_model = options["use-neural-trained-model"] == "true"

            if (options["inference_split"] == "VALID") and use_neural_trained_model:
                self.evaluation_out_dir = os.path.join(BASE_RESULTS_DIR, options["task-name"], NEUPSL_VALID_INFERENCE_DIR,
                                                       NEURAL_TRAINED_MODEL_DIR, options["evaluation-dir-name"])
            elif (options["inference_split"] == "VALID") and (not use_neural_trained_model):
                self.evaluation_out_dir = os.path.join(BASE_RESULTS_DIR, options["task-name"], NEUPSL_VALID_INFERENCE_DIR,
                                                       NEUPSL_TRAINED_MODEL_DIR, options["evaluation-dir-name"])
            elif (options["inference_split"] == "TEST") and use_neural_trained_model:
                self.evaluation_out_dir = os.path.join(BASE_RESULTS_DIR, options["task-name"], NEUPSL_TEST_INFERENCE_DIR,
                                                       NEURAL_TRAINED_MODEL_DIR, options["evaluation-dir-name"])
            elif (options["inference_split"] == "TEST") and (not use_neural_trained_model):
                self.evaluation_out_dir = os.path.join(BASE_RESULTS_DIR, options["task-name"], NEUPSL_TEST_INFERENCE_DIR,
                                                       NEUPSL_TRAINED_MODEL_DIR, options["evaluation-dir-name"])
            else:
                raise ValueError("Invalid inference split: {0}".format(options["inference_split"]))

            if use_neural_trained_model:
                trained_model_path = neural_trained_model_path
            else:
                trained_model_path = os.path.join(self.model_out_dir, NEUPSL_TRAINED_MODEL_FILENAME)

            logging.info("Loading model from: {0}".format(trained_model_path))

            if options["load-predictions"] == "false":
                self.model.load_state_dict(torch.load(trained_model_path, map_location=get_torch_device()))

            annotation_path = TRAIN_VALIDATION_DATA_PATH if options["inference_split"] == "VALID" else None
            self.dataset = RoadRDataset(VIDEO_PARTITIONS[options["task-name"]][options["inference_split"]], annotation_path, float(options["image-resize"]),
                                        max_frames=int(options["max-frames"]), test=(options["inference_split"] == "TEST"))
            self.optimizer = None
            self.scheduler = None

        self.dataloader = DataLoader(self.dataset, batch_size=int(options["batch-size"]),
                                     shuffle=False, num_workers=int(os.cpu_count()) - 2,
                                     prefetch_factor=4, persistent_workers=True)

        return {}

    def internal_fit(self, data, gradients, options={}):
        structured_gradients = float(options["alpha"]) * self.format_batch_gradients(gradients, len(self.current_batch[1]))

        self.optimizer.zero_grad(set_to_none=True)
        self.batch_predictions["logits"].backward(structured_gradients, retain_graph=True)

        frame_ids, pixel_values, pixel_mask, labels, boxes = self.gpu_batch
        loss, results = detr_loss(self.batch_predictions["pred_boxes"], self.batch_predictions["logits"], boxes, labels)

        loss = (1 - float(options["alpha"])) * loss
        loss.backward(retain_graph=True)

        self.post_gradient_computation()
        self.optimizer.step()

        return results

    def internal_predict(self, data, options={}):
        frame_indexes, pixel_values, pixel_mask, labels, boxes = self.gpu_batch

        if options["load-predictions"] == "true":
            frame_ids = []
            for frame_index in frame_indexes:
                frame_ids.append(self.dataset.get_frame_id(frame_index))
            self.batch_predictions = self.model.load_predictions(frame_ids)
        else:
            self.batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})

        if (self.application == "inference") and (not self._train):
            self.all_frame_indexes.extend(frame_indexes.cpu().tolist())

        return self.format_batch_predictions(self.batch_predictions["logits"].detach().cpu(), self.batch_predictions["pred_boxes"].detach().cpu(), options=options), {}

    def internal_eval(self, data, options={}):
        if self.application == "learning":
            return 0.0

        self.format_batch_results(options=options)

        return 0.0

    def internal_epoch_start(self, options={}):
        self.all_box_predictions = []
        self.all_class_predictions = []
        self.all_frame_indexes = []

        self.iterator = iter(self.dataloader)
        self.next_batch()

    def internal_epoch_end(self, options={}):
        if (self.application == "learning") and (self._train):
            self.scheduler.step()
            self.save()
        elif (self.application == "inference") and (not self._train):
            os.makedirs(self.evaluation_out_dir, exist_ok=True)
            save_logits_and_labels(self.dataset, self.all_frame_indexes, self.all_class_predictions, self.all_box_predictions,
                                   output_dir=self.evaluation_out_dir, from_logits=False)

            if options["inference_split"] == "VALID":
                calculate_metrics(self.dataset, self.evaluation_out_dir)

            save_images_with_bounding_boxes(self.dataset, self.evaluation_out_dir, True,
                                            NUM_SAVED_IMAGES, LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD,
                                            write_ground_truth=(options["inference_split"] == "VALID"),
                                            test=(options["inference_split"] == "TEST"))

    def internal_next_batch(self, options={}):
        self.current_batch = next(self.iterator, None)
        if self.current_batch is not None:
            self.gpu_batch = [b.to(get_torch_device()) for b in self.current_batch]

    def internal_is_epoch_complete(self, options={}):
        if self.current_batch is None:
            return True
        return False

    def internal_save(self, options={}):
        os.makedirs(self.model_out_dir, exist_ok=True)
        save_model_state(self.model, self.model_out_dir, NEUPSL_TRAINED_MODEL_FILENAME)

    def train_mode(self, options = {}):
        super().train_mode(options=options)
        self.model.train()

    def eval_mode(self, options = {}):
        super().eval_mode(options=options)
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

        return formatted_gradients.to(get_torch_device())

    def format_batch_predictions(self, logits, boxes, options={}):
        self.batch_predictions_formatted = torch.zeros(size=(int(options["batch-size"]), NUM_NEUPSL_QUERIES, int(options["class-size"])), dtype=torch.float32)

        batch_predictions_sorted_indexes = torch.argsort(logits[:, :, -1], descending=True)

        batch_prediction_probabilities = torch.sigmoid(logits)
        batch_predictions = torch.cat((batch_prediction_probabilities,
                                       torch.zeros(size=boxes.shape, dtype=torch.float32)), dim=-1)

        for batch_index in range(len(batch_prediction_probabilities)):
            for box_index in range(NUM_NEUPSL_QUERIES):
                if (self.application == "inference") and (batch_prediction_probabilities[batch_index][batch_predictions_sorted_indexes[batch_index][box_index]][-1]) < BOX_CONFIDENCE_THRESHOLD:
                    continue

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

    def post_gradient_computation(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)


class LoadPredictionsModel:
    def __init__(self, predictions_path):
        self.predictions_path = predictions_path
        self.predictions = load_json_file(self.predictions_path)

    def load_predictions(self, frame_ids):
        frame_predictions = {"logits": [], "pred_boxes": []}
        for batch_frame_index in range(len(frame_ids)):
            frame_id = frame_ids[batch_frame_index]
            frame_predictions["logits"].append([])
            frame_predictions["pred_boxes"].append([])
            for box in range(len(self.predictions[frame_id[0]][frame_id[1]])):
                frame_predictions["logits"][batch_frame_index].append(self.predictions[frame_id[0]][frame_id[1]][box]["labels"])
                frame_predictions["pred_boxes"][batch_frame_index].append(self.predictions[frame_id[0]][frame_id[1]][box]["bbox"])

        frame_predictions["logits"] = torch.torch.tensor(frame_predictions["logits"], dtype=torch.float32)
        frame_predictions["logits"] = torch.log(frame_predictions["logits"] / (1 - frame_predictions["logits"]))
        frame_predictions["pred_boxes"] = torch.tensor(frame_predictions["pred_boxes"], dtype=torch.float32)
        return frame_predictions

    def eval(self):
        pass

    def train(self):
        pass
