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

import data.RoadRDataset as RoadRDataset

from models.detr import DETR
from models.losses import binary_cross_entropy
from models.hungarian_matcher import HungarianMatcher


class RoadRDETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    """
    DETR Model from: <https://arxiv.org/pdf/2005.12872.pdf>
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.hungarianMatcher = None
        self.predictions = None
        self.training_data = None

    def internal_init_model(self, application, options={}):
        resnet50 = torchvision.models.resnet50(weights=None)

        # Remove the last two layers of the resnet50 model.
        # The last layer is a fully connected layer and the second to last layer is a pooling layer.
        backbone = torch.nn.Sequential(*list(resnet50.children())[:-2])

        transformer = torch.nn.Transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.model = DETR(backbone, transformer)

        # Initialize the matcher for the loss function.
        self.hungarianMatcher = HungarianMatcher()

        return {}

    def internal_fit(self, data, gradients, options={}):
        loss(self.predictions, data)
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

        if options["learn"]:
            # Compute the training loss.
            # For the training loss, we need to first compute the matching between the predictions and the ground truth.
            matching = self.hungarianMatcher(self.predictions["boxes"], boxes)

            # Compute the classification loss using the matching.
            bce_loss = binary_cross_entropy(self.predictions["class_probabilities"], labels, matching)

            # Compute the bounding box loss using the matching.


        return {}

    def internal_save(self, options={}):
        return {}
