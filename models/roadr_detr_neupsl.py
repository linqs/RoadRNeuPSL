#!/usr/bin/env python3
import logging
import os
import sys

import pslpython.deeppsl.model
import torch.nn
import torchvision

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logger
import utils

from data.roadr_dataset import RoadRDataset
from models.detr import DETR
from models.losses import binary_cross_entropy
from models.losses import pairwise_generalized_box_iou
from models.hungarian_match import hungarian_match

logger.initLogging(logging_level = logging.DEBUG)

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "..", "data")
DATA_FILE_NAME = 'road_trainval_v1.0.json'

LABELED_VIDEOS = ["2014-07-14-14-49-50_stereo_centre_01",
                  "2015-02-03-19-43-11_stereo_centre_04",
                  "2015-02-24-12-32-19_stereo_centre_04"]

MAX_FRAMES = 40
TUBE_SIZE = 4

class RoadRDETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    """
    DETR Model from: <https://arxiv.org/pdf/2005.12872.pdf>
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.predictions = None
        self.training_data = None
        self.current_batch = None

        self.bce_weight = 1
        self.giou_weight = 2

        self.dataset = RoadRDataset(LABELED_VIDEOS, os.path.join(DATA_DIR, DATA_FILE_NAME), max_frames=MAX_FRAMES)

        self.train_dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        self.train_iterator = iter(self.train_dataloader)

    def internal_init_model(self, application, options={}):
        logging.info("Initializing neural model for application: {0}".format(application))

        resnet34 = torchvision.models.resnet34(weights=None)

        # Remove the last two layers of the resnet34 model.
        # The last layer is a fully connected layer and the second to last layer is a pooling layer.
        backbone = torch.nn.Sequential(*list(resnet34.children())[:-2])

        transformer = torch.nn.Transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=512,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.model = DETR(backbone, transformer).to(utils.get_torch_device())

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)

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
        self.current_batch = next(self.train_iterator)
        frames, train_images, labels, boxes = self.current_batch

        self.predictions = self.model(train_images)
        predictions = torch.cat((self.predictions["class_probabilities"], self.predictions["boxes"]), dim=2)
        predictions = predictions.view(-1, predictions.shape[-1])

        # TODO(Connor): Predictions dimension is (100, 46), but should be (100, 45)
        return predictions.tolist(), {}

    def internal_eval(self, data, options={}):
        results = {}

        if options["learn"]:
            results = self._compute_loss()

        return results

    def internal_save(self, options={}):
        return {}

    def _compute_loss(self):
        results = {}

        frames, train_images, labels, boxes = self.current_batch

        # Compute the matching between the predictions and the ground truth.
        matching = hungarian_match(self.predictions["boxes"], boxes)

        # Compute the classification loss using the matching.
        results["bce_loss"] = binary_cross_entropy(self.predictions["class_probabilities"], labels, matching)

        # Compute the bounding box loss using the matching.
        results["giou_loss"] = pairwise_generalized_box_iou(self.predictions["boxes"], boxes, matching)

        return results
