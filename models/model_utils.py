"""
Utilities for RoadRNeuPSL models.
"""
import os
import re
import torch

from utils import TRAINED_MODEL_FILENAME


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box format from [x_c, y_c, w, h] to [x0, y0, x1, y1]
    where x_c, y_c are the center coordinates, w, h are the width and height of the box.
    :param x: The bounding box in [x_c, y_c, w, h] format.
    :return: The bounding box in [x0, y0, x1, y1] format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding box format from [x0, y0, x1, y1] to [x_c, y_c, w, h]
    where x_c, y_c are the center coordinates, w, h are the width and height of the box.
    :param x: The bounding box in [x0, y0, x1, y1] format.
    :return: The bounding box in [x_c, y_c, w, h] format.
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def load_model_state(model: torch.nn.Module, out_directory: str):
    model.load_state_dict(torch.load(os.path.join(out_directory, TRAINED_MODEL_FILENAME)), strict=True)


def save_model_state(model: torch.nn.Module, out_directory: str):
    formatted_model_state_dict = {
        re.sub(r"^module\.", "", key).strip(): model.state_dict()[key]
        for key in model.state_dict()
    }
    torch.save(formatted_model_state_dict, os.path.join(out_directory, TRAINED_MODEL_FILENAME))
