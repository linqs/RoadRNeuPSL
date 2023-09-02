import os
import sys

import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils

from models.detr import DETR


def build_task_1_model(dropout=0.1, image_resize=1.0, num_queries=20):
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
        dropout=dropout,
        activation='relu',
        batch_first=True,
        norm_first=False
    )
    return DETR(backbone, transformer, image_resize=image_resize, num_queries=num_queries).to(utils.get_torch_device())
