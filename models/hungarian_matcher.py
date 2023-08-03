# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from models.losses import generalized_box_iou
from models.model_utils import box_cxcywh_to_xyxy


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_bbox: float = 0, cost_giou: float = 2):
        """
        Creates the matcher

        Params:
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, pred_boxes, truth_boxes):
        """
        Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "class_probabilities": Tensor of dim [batch_size, num_queries, num_classes] with the classification probabilities
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            truth_boxes: This is a list of targets bounding boxes (len(targets) = batch_size), where each entry is
                a tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = pred_boxes.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target boxes
        tgt_bbox = truth_boxes.flatten(0, 1)  # [batch_size * num_target_boxes, 4]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v) for v in truth_boxes]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return torch.stack([torch.stack((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))) for i, j in indices])
