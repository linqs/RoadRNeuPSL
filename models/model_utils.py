# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch


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