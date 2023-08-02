#!/usr/bin/env python3
import os
import sys

import numpy
import torchvision

import pslpython.deeppsl.model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logger
import utils

import data.RoadRDataset as RoadRDataset
from models.losses import simple_loss as loss
from models.DETR import DETR


"""
DETR Model from: <https://arxiv.org/pdf/2005.12872.pdf>
"""
class DETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self._model = None
        self._predictions = None

    def internal_init_model(self, application, options={}):
        self._model = DETR()
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

        # TODO(Charles): Model prediction requires 'targets' in training mode.
        #  This is the target bounding boxes for the image.
        #  Targets are required because the fcos model computes a loss during prediction in the torchvision implementation.
        self._predictions = self._model(data)
        return self._predictions, results

    def internal_eval(self, data, options={}):
        return {}

    def internal_save(self, options={}):
        return {}
