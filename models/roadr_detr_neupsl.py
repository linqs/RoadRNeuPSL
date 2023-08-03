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

from models.losses import simple_loss as loss
from models.DETR import DETR


class RoadRDETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    """
    DETR Model from: <https://arxiv.org/pdf/2005.12872.pdf>
    """
    def __init__(self):
        super().__init__()
        self._model = None
        self._predictions = None

    def internal_init_model(self, application, options={}):
        backbone = torchvision.models.resnet50(weights=None)

        transformer = torch.nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            norm_first=False
        )
        self._model = DETR(backbone, transformer)
        return {}

    def internal_fit(self, data, gradients, options={}):
        loss(self._predictions, data)
        return {}

    def internal_predict(self, data, options={}):
        results = {}
        if options["learn"]:
            results['mode'] = 'learning'
            self._model.train()
        else:
            results['mode'] = 'inference'
            self._model.eval()

        self._predictions = self._model(data)

        return self._predictions, results

    def internal_eval(self, data, options={}):
        return {}

    def internal_save(self, options={}):
        return {}
