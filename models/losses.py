import logging
import os

import torch
import torchvision

from scipy.optimize import linear_sum_assignment

from utils import write_json_file
from utils import BASE_RESULTS_DIR

BCE_WEIGHT = 1
GIOU_WEIGHT = 5
L1_WEIGHT = 2


def detr_loss(pred_boxes, pred_logits, truth_boxes, truth_labels, bce_weight: int = BCE_WEIGHT, giou_weight: int = GIOU_WEIGHT, l1_weight: int = L1_WEIGHT, model=None):
    """
    Computes the detr loss for the given outputs and targets.
    First compute the matching between the predictions and the ground truth using hungarian sort.
    Then compute the classification loss, pairwise generalized box iou loss, and pairwise l1 loss using this matching.
    Finally, compute the total loss as a weighted sum of the three losses.
    :param pred_boxes: Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
    :param pred_logits: Tensor of dim [batch_size, num_queries, num_classes] with predicted class logits
    :param truth_boxes: This is a list of targets bounding boxes (len(targets) = batch_size), where each entry is
            a tensor of dim [num_target_boxes, 4] containing the target box coordinates
    :param truth_labels: Tensor of dim [batch_size, num_queries, num_classes] with target class labels
    :param bce_weight: Weight for the binary cross entropy loss
    :param giou_weight: Weight for the generalized box iou loss
    :param l1_weight: Weight for the l1 loss
    :param model: (Optional) The model used for predictions. Providing this will print model parameters when NaNs are detected.
    :return: Detr loss and a dict containing the computed losses
    """
    # First compute the matching between the predictions and the ground truth.
    matching = _hungarian_match(pred_boxes, truth_boxes, model=model)

    # Compute the classification loss using the matching.
    bce_loss = binary_cross_entropy_with_logits(pred_logits, truth_labels, matching)

    # Compute the bounding box loss using the matching.
    giou_loss = pairwise_generalized_box_iou(pred_boxes, truth_boxes, matching, model=model)

    # Compute the bounding box l2 loss using the matching.
    l1_loss = pairwise_l1_loss(pred_boxes, truth_boxes, matching)

    results = {
        "bce_loss": bce_loss.item(),
        "bce_weight": bce_weight,
        "giou_loss": giou_loss.item(),
        "giou_weight": giou_weight,
        "l1_loss": l1_loss.item(),
        "l1_weight": l1_weight,
        "loss": (bce_weight * bce_loss + giou_weight * giou_loss + l1_weight * l1_loss).item()
    }

    return bce_weight * bce_loss + giou_weight * giou_loss + l1_weight * l1_loss, results


def binary_cross_entropy_with_logits(outputs, truth, indices) -> torch.Tensor:
    """
    Computes the binary cross entropy loss for the given outputs and targets.
    The targets are aligned with the outputs using the given indices.
    :param outputs: the outputs of the model. dict of tensors of shape [batch_size, num_queries, num_classes]
    :param truth: the ground truth labels. tensor of shape [batch_size, num_queries, num_classes]
    :param indices: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
    :return: the computed binary cross entropy loss
    """
    # Align the outputs and targets using the given indices and flatten.
    aligned_outputs = torch.stack([outputs[i, indices[i, 0, :], :] for i in range(outputs.shape[0])]).flatten(0, 1)
    aligned_truth = torch.stack([truth[i, indices[i, 1, :], :] for i in range(truth.shape[0])]).flatten(0, 1).to(torch.float32)

    return torch.nn.functional.binary_cross_entropy_with_logits(aligned_outputs, aligned_truth)


def pairwise_l1_loss(boxes1, boxes2, indices) -> torch.Tensor:
    """
    Computes the pairwise l1 loss for the given outputs and targets aligned using the given indices.
    :param boxes1: The first set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param boxes2: The second set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param indices: The indices used to align the boxes. A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
    :return: The computed pairwise l1 loss.
    """
    # Align the boxes using the given indices.
    aligned_boxes1 = torch.stack([boxes1[i, indices[i, 0, :], :] for i in range(boxes1.shape[0])]).flatten(0, 1)
    aligned_boxes2 = torch.stack([boxes2[i, indices[i, 1, :], :] for i in range(boxes2.shape[0])]).flatten(0, 1)

    return torch.nn.functional.l1_loss(aligned_boxes1, aligned_boxes2)


def pairwise_generalized_box_iou(boxes1, boxes2, indices, model=None) -> torch.Tensor:
    """
    Computes the pairwise generalized box iou loss for the given outputs and targets aligned using the given indices.
    :param boxes1: The first set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param boxes2: The second set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param indices: The indices used to align the boxes. A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
    :param model: (Optional) The model used for predictions. Providing this will print model parameters when NaNs are detected.
    :return: The computed pairwise generalized box iou loss.
    """
    # Align the boxes using the given indices.
    aligned_boxes1 = torch.stack([boxes1[i, indices[i, 0, :], :] for i in range(boxes1.shape[0])]).flatten(0, 1)
    aligned_boxes2 = torch.stack([boxes2[i, indices[i, 1, :], :] for i in range(boxes2.shape[0])]).flatten(0, 1)

    return (1 - torch.diag(generalized_box_iou(aligned_boxes1, aligned_boxes2, model=model)).sum() / aligned_boxes1.shape[0])


def generalized_box_iou(boxes1, boxes2, model=None):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    :param boxes1: Tensor of shape [N, 4]
    :param boxes2: Tensor of shape [M, 4]
    :param model: (Optional) The model used for predictions. Providing this will print model parameters when NaNs are detected.
    :return: A [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    if torch.isnan(boxes1).any():
        logging.info("Boxes1 contains NaNs.")
        log = {"boxes": boxes1.tolist(), "model": {}}
        if model is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    log["model"][name] = param.data.tolist()
        write_json_file(os.path.join(BASE_RESULTS_DIR, "nan_boxes1.json"), log, indent=None)
        assert not torch.isnan(boxes1).any()
    if torch.isnan(boxes2).any():
        logging.info("Boxes2 contains NaNs.")
        log = {"boxes": boxes2.tolist(), "model": {}}
        if model is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    log["model"][name] = param.data.tolist()
        write_json_file(os.path.join(BASE_RESULTS_DIR, "nan_boxes2.json"), log, indent=None)
        assert not torch.isnan(boxes1).any()

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    """
    Compute the intersection over union of two set of boxes, each box is [x0, y0, x1, y1].
    This is modified from torchvision to also return the union.
    :param boxes1: Tensor of shape [N, 4]
    :param boxes2: Tensor of shape [M, 4]
    :return: A [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    area1 = torchvision.ops.boxes.box_area(boxes1)
    area2 = torchvision.ops.boxes.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def single_box_iou(box1, box2):
    """
    Compute the intersection over union of two boxes, each box is [x0, y0, x1, y1].
    :param box1: Tensor of shape [4]
    :param box2: Tensor of shape [4]
    :return: A scalar representing the iou of the two boxes.
    """
    area1 = torchvision.ops.boxes.box_area(box1)
    area2 = torchvision.ops.boxes.box_area(box2)

    lt = torch.max(box1[:, :2], box2[:, :2])  # [2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [2]

    wh = (rb - lt).clamp(min=0)  # [2]
    inter = wh[:, 0] * wh[:, 1]  # [1]

    union = area1 + area2 - inter

    iou = inter / union
    return iou


def _hungarian_match(pred_boxes, truth_boxes, l1_weight: int=0, giou_weight: int=1, model=None):
    """
    Computes an assignment between the predictions and the truth boxes.
    :param pred_boxes: Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates in [x0, y0, x1, y1] format.
    :parm truth_boxes: This is a list of targets bounding boxes (len(targets) = batch_size), where each entry is
            a tensor of dim [num_target_boxes, 4] containing the target box coordinates in [x0, y0, x1, y1] format.
    :param model: (Optional) The model used for predictions. Providing this will print model parameters when NaNs are detected.
    :return: A list of size batch_size, containing tuples of (index_i, index_j) where:
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
    cost_giou = -generalized_box_iou(out_bbox, tgt_bbox, model=model)

    # Final cost matrix
    C = l1_weight * cost_bbox + giou_weight * cost_giou
    C = C.view(batch_size, num_queries, -1).cpu()

    sizes = [len(v) for v in truth_boxes]
    indices = [linear_sum_assignment(c[i].detach().numpy()) for i, c in enumerate(C.split(sizes, -1))]

    return torch.stack([torch.stack((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))) for i, j in indices])
