import sys

import numpy
import torch

from torchvision.ops import nms

# CONFIDENCE_THRESHOLD = 0.025
CONFIDENCE_THRESHOLD = 0.000

CLASS_LABEL_TYPE = ["agent"] * 10 + ["action"] * 19 + ["loc"] * 12


def mean_average_precision(ground_truths, predictions, iou_threshold=0.5):
    """
    Computes the mean average precision (mAP) for a given set of predictions and ground truths.
    :param ground_truths: Dictionary containing the ground truths for each label type.
    :param predictions: Dictionary containing the predictions for each label type.
    :param iou_threshold: Threshold for the IoU.
    """

    average_precisions = []

    # for label_type in LABEL_TYPES:
    #     average_precision = compute_average_precision(ground_truths[label_type], predictions[label_type], iou_threshold)
    #     average_precisions.append(average_precision)

    return numpy.mean(average_precisions)


def load_labels(dataset, detections):
    """
    Loads the ground truth labels and formats the same as the detections.
    :param dataset: Dataset for which the labels should be loaded.
    :param detections: Dictionary containing the detections for each frame.
    """

    ground_truth = {}

    for frame_index in detections.keys():
        ground_truth[frame_index] = {}

        frame_truth_labels = dataset[dataset.frame_ids_mapping[int(frame_index)]][2]
        frame_truth_boxes = dataset[dataset.frame_ids_mapping[int(frame_index)]][3]

        print(frame_truth_labels.shape, frame_truth_labels)

        for class_index in sorted(detections[frame_index].keys()):
            frame_truth_label = frame_truth_labels[:, class_index]


            # mask_frame_scores = frame_scores.gt(CONFIDENCE_THRESHOLD)
            #
            # frame_boxes = frame_boxes[mask_frame_scores]
            # frame_scores = frame_scores[mask_frame_scores]
            #
            # if class_index not in ground_truth[frame_index]:
            #     ground_truth[frame_index][class_index] = torch.Tensor([])
            # ground_truth[frame_index][class_index] = torch.cat((frame_truth_boxes, frame_truth_labels[class_index].unsqueeze(1)), dim=1)
        sys.exit(0)

    return ground_truth


def filter_detections(frame_indexes: torch.Tensor, box_predictions: torch.Tensor, class_predictions: torch.Tensor, iou_threshold: float):
    """
    Filters detections by first removing all detections with a confidence score below a threshold and then applying non-maximum suppression.
    :param frame_indexes: List of frame indexes (N, 2) for which the detections were computed.
    :param box_predictions: Tensor of shape (N, 4) containing the predicted bounding boxes.
    :param class_predictions: Tensor of shape (N, C) containing the predicted class probabilities.
    :param iou_threshold: Threshold for the IoU.
    """

    filtered_detections = {}

    for frame_index in range(frame_indexes.shape[0]):
        frame_id = frame_indexes[frame_index]
        frame_boxes = box_predictions[frame_index]

        filtered_detections[frame_id] = {}

        for class_index in range(class_predictions.shape[2]):
            frame_scores = class_predictions[frame_index, :, class_index]

            mask_frame_scores = frame_scores.gt(CONFIDENCE_THRESHOLD)

            frame_boxes = frame_boxes[mask_frame_scores]
            frame_scores = frame_scores[mask_frame_scores]

            kept_element_indexes = nms(boxes=frame_boxes, scores=frame_scores, iou_threshold=iou_threshold)

            if len(kept_element_indexes) == 0:
                filtered_detections[frame_id][class_index] = torch.Tensor([])
                continue

            filtered_frame_boxes = frame_boxes[kept_element_indexes]
            filtered_frame_scores = frame_scores[kept_element_indexes]

            filtered_detections[frame_id][class_index] = torch.cat((filtered_frame_boxes, filtered_frame_scores.unsqueeze(1)), dim=1)

    return filtered_detections
