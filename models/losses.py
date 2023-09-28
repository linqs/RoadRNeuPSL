import torch
import torchvision

from models.model_utils import box_cxcywh_to_xyxy


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


def pairwise_l2_loss(boxes1, boxes2, indices) -> torch.Tensor:
    """
    Computes the pairwise l2 loss for the given outputs and targets aligned using the given indices.
    :param boxes1: The first set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param boxes2: The second set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param indices: The indices used to align the boxes. A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
    :return: The computed pairwise l2 loss.
    """
    # Align the boxes using the given indices.
    aligned_boxes1 = torch.stack([boxes1[i, indices[i, 0, :], :] for i in range(boxes1.shape[0])]).flatten(0, 1)
    aligned_boxes2 = torch.stack([boxes2[i, indices[i, 1, :], :] for i in range(boxes2.shape[0])]).flatten(0, 1)

    return torch.nn.functional.mse_loss(aligned_boxes1, aligned_boxes2)


def pairwise_generalized_box_iou(boxes1, boxes2, indices) -> torch.Tensor:
    """
    Computes the pairwise generalized box iou loss for the given outputs and targets aligned using the given indices.
    :param boxes1: The first set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param boxes2: The second set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param indices: The indices used to align the boxes. A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
    :return: The computed pairwise generalized box iou loss.
    """
    # Align the boxes using the given indices.
    aligned_boxes1 = torch.stack([boxes1[i, indices[i, 0, :], :] for i in range(boxes1.shape[0])]).flatten(0, 1)
    aligned_boxes2 = torch.stack([boxes2[i, indices[i, 1, :], :] for i in range(boxes2.shape[0])]).flatten(0, 1)

    return (1 - torch.diag(generalized_box_iou(
        box_cxcywh_to_xyxy(aligned_boxes1),
        box_cxcywh_to_xyxy(aligned_boxes2))).sum() / aligned_boxes1.shape[0])


def box_iou(boxes1, boxes2):
    # modified from torchvision to also return the union
    area1 = torchvision.ops.boxes.box_area(boxes1)
    area2 = torchvision.ops.boxes.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
