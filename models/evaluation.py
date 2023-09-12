import torch

from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms

CONFIDENCE_THRESHOLD = 0.025
NUM_CLASSES = 41


def filter_detections(frame_indexes, box_predictions, class_predictions, iou_threshold, num_classes=NUM_CLASSES):
    """
    Filters detections by first removing all detections with a confidence score below a threshold and then applying non-maximum suppression.
    :param frame_indexes: List of frame indexes (N, 2) for which the detections were computed.
    :param box_predictions: Tensor of shape (N, 4) containing the predicted bounding boxes.
    :param class_predictions: Tensor of shape (N, C) containing the predicted class probabilities.
    :param iou_threshold: Threshold for the IoU.
    :param num_classes: Number of classes in the dataset.
    """
    filtered_detections = []
    filtered_detection_indexes = []

    for frame_index in range(frame_indexes.shape[0]):
        frame_boxes = box_predictions[frame_index]

        filtered_detection_indexes.append(int(frame_indexes[frame_index]))

        for class_index in range(num_classes):
            frame_scores = class_predictions[frame_index, :, class_index]

            mask_frame_scores = frame_scores.gt(CONFIDENCE_THRESHOLD)

            masked_frame_boxes = frame_boxes[mask_frame_scores]
            masked_frame_scores = frame_scores[mask_frame_scores]

            kept_element_indexes = nms(boxes=masked_frame_boxes, scores=masked_frame_scores, iou_threshold=iou_threshold)

            if len(kept_element_indexes) == 0:
                filtered_detections.append({'boxes': torch.Tensor([]), 'scores': torch.Tensor([]), 'labels': torch.Tensor([])})
                continue

            filtered_frame_boxes = masked_frame_boxes[kept_element_indexes]
            filtered_frame_scores = masked_frame_scores[kept_element_indexes]
            filtered_frame_labels = torch.Tensor([int(class_index)] * len(filtered_frame_boxes)).type(torch.int64)

            filtered_detections.append({'boxes': filtered_frame_boxes, 'scores': filtered_frame_scores, 'labels': filtered_frame_labels})

    return filtered_detections, filtered_detection_indexes


def load_ground_truth_for_detections(dataset, indexes, num_classes=NUM_CLASSES):
    """
    Loads the ground truth labels for the given detections.
    :param dataset: Dataset for which the labels should be loaded.
    :param indexes: List of indexes for which the detections were computed.
    :param num_classes: Number of classes in the dataset.
    """
    ground_truth = []

    for frame_index in indexes:
        frame_truth_boxes = dataset[frame_index][3]
        frame_truth_labels = dataset[frame_index][2]

        mask_frame_truth_boxes = frame_truth_boxes.sum(dim=1).gt(0)

        frame_truth_boxes = frame_truth_boxes[mask_frame_truth_boxes]
        frame_truth_labels = frame_truth_labels[mask_frame_truth_boxes]

        for class_index in range(num_classes):
            class_frame_truth_labels = frame_truth_labels[:, class_index]

            mask_class_frame_truth_labels = class_frame_truth_labels.gt(0)

            if class_frame_truth_labels.sum() == 0:
                ground_truth.append({'boxes': torch.Tensor([]), 'labels': torch.Tensor([])})
                continue

            class_frame_truth_boxes = frame_truth_boxes[mask_class_frame_truth_labels]
            class_frame_frame_labels = torch.Tensor([int(class_index)] * len(class_frame_truth_boxes)).type(torch.int64)

            ground_truth.append({'boxes': class_frame_truth_boxes, 'labels': class_frame_frame_labels})

    return ground_truth


def mean_average_precision(ground_truths, detections, iou_threshold=0.5):
    """
    Computes the mean average precision (mAP) for a given set of predictions and ground truths.
    :param ground_truths: List containing the ground truths.
    :param detections: List containing the detections.
    :param iou_threshold: Threshold for the IoU.
    """
    map_metric = MeanAveragePrecision(iou_thresholds=[iou_threshold], iou_type="bbox")

    map_metric.update(detections, ground_truths)
    values = map_metric.compute()

    return float(values['map'])
