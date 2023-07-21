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


class FCOSNeuPSL(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self._application = None
        self._model = None
        self._predictions = None

    def internal_init_model(self, application, options={}):
        self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=RoadRDataset.NUM_CLASSES)
        return {}

    def internal_fit(self, data, gradients, options={}):
        loss(self._predictions, data)
        return {}

    def internal_predict(self, data, options={}):
        if self._application == 'learning':
            self._model.train()
        else:
            self._model.eval()

        self._predictions = self._model([data[0][1]])
        return {}

    def internal_eval(self, data, options={}):
        return {}

    def internal_save(self, options={}):
        return {}
