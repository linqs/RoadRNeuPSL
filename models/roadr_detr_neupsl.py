#!/usr/bin/env python3
import os
import sys

import numpy
import torch.nn
import torchvision

import pslpython.deeppsl.model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logger
import utils

from models.detr import DETR
from models.losses import binary_cross_entropy
from models.losses import pairwise_generalized_box_iou
from models.hungarian_match import hungarian_match


class RoadRDETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    """
    DETR Model from: <https://arxiv.org/pdf/2005.12872.pdf>
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.predictions = None
        self.training_data = None

    def internal_init_model(self, application, options={}):
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

        return {}

    def internal_fit(self, data, gradients, options={}):
        return {}

    def internal_predict(self, data, options={}):
        results = {}
        if options["learn"]:
            results['mode'] = 'learning'
            self.model.train()
        else:
            results['mode'] = 'inference'
            self.model.eval()

        self.predictions = self.model(data)

        return self.predictions, results

    def internal_eval(self, data, options={}):
        frames, train_images, labels, boxes = data

        results = {}
        if options["learn"]:
            # Compute the training loss.
            # For the training loss, we need to first compute the matching between the predictions and the ground truth.
            matching = hungarian_match(self.predictions["boxes"], boxes)

            # Compute the classification loss using the matching.
            bce_loss = binary_cross_entropy(self.predictions["class_probabilities"], labels, matching)

            # Compute the bounding box loss using the matching.
            giou_loss = pairwise_generalized_box_iou(self.predictions["boxes"], boxes, matching)

            results["bce_loss"] = bce_loss
            results["giou_loss"] = giou_loss

        return results

    def internal_save(self, options={}):
        return {}
