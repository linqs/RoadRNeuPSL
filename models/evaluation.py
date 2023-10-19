import numpy as np
import torch

from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms

from models.losses import single_box_iou
from utils import pixel_to_ratio_coordinates
from utils import ratio_to_pixel_coordinates
from utils import AGENT_CLASSES
from utils import NUM_CLASSES

EPSILON = 1e-7


def agent_nms(box_predictions, box_confidence_scores, class_predictions, iou_threshold=0.6, label_threshold=0.5):
    """
    Applies non-maximum suppression to the detections using the agent class.
    :param box_predictions: The predicted bounding boxes.
    :param box_confidence_scores: The confidence scores for the predicted bounding boxes.
    :param class_predictions: The predicted class labels in one-hot encoding.
    :param iou_threshold: Threshold for the IoU.
    :param label_threshold: Threshold for the label confidence score.
    :return: Indices of the detections that should be kept.
    """
    kept_element_indexes = []
    for frame_index in range(box_predictions.shape[0]):
        frame_boxes = box_predictions[frame_index]
        frame_scores = box_confidence_scores[frame_index]

        kept_element_indexes.append(set())

        for class_index in AGENT_CLASSES:
            class_scores = class_predictions[frame_index, :, class_index]

            class_frames_mask = class_scores > label_threshold

            masked_frame_boxes = frame_boxes[class_frames_mask]
            masked_frame_scores = frame_scores[class_frames_mask]

            nms_kept_indices = nms(boxes=torch.tensor(masked_frame_boxes), scores=torch.tensor(masked_frame_scores), iou_threshold=iou_threshold)
            nms_kept_indices = nms_kept_indices.numpy()
            nms_kept_indices = np.arange(len(frame_boxes))[class_frames_mask][nms_kept_indices]

            for index in nms_kept_indices:
                kept_element_indexes[-1].add(index)

        kept_element_indexes[-1] = list(kept_element_indexes[-1])

    return kept_element_indexes


def filter_map_pred_and_truth(dataset, frame_indexes, box_predictions, class_predictions, iou_threshold, confidence_threshold, num_classes=NUM_CLASSES):
    """
    Filters predictions using non-maximum suppression and returns the predictions and truth in the format
    required for the mean average precision metric.
    :param dataset: Dataset for which the detections were computed.
    :param box_predictions: Tensor of shape (N, 4) containing the predicted bounding boxes.
    :param class_predictions: Tensor of shape (N, C) containing the predicted class probabilities.
    :param iou_threshold: Threshold for the IoU.
    :param confidence_threshold: Threshold for box and class confidence.
    :param num_classes: Number of classes in the dataset.
    :return: Filtered predictions and truth.
    """
    filtered_pred = []
    filtered_truth = []

    for frame_index in frame_indexes:
        frame_pred_labels, frame_pred_boxes = class_predictions[frame_index], box_predictions[frame_index]
        frame_truth_labels, frame_truth_boxes = dataset.get_labels_and_boxes(frame_index)

        mask_frame_pred = frame_pred_labels[:, -1].gt(confidence_threshold)
        mask_frame_truth = frame_truth_boxes.sum(dim=1).gt(EPSILON)

        frame_pred_labels, frame_pred_boxes = frame_pred_labels[mask_frame_pred], frame_pred_boxes[mask_frame_pred]
        frame_truth_labels, frame_truth_boxes = frame_truth_labels[mask_frame_truth], frame_truth_boxes[mask_frame_truth]

        frame_pred_boxes = ratio_to_pixel_coordinates(frame_pred_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)
        frame_truth_boxes = ratio_to_pixel_coordinates(frame_truth_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)

        for class_index in range(num_classes):
            class_pred_labels = frame_pred_labels[:, class_index]
            class_truth_labels = frame_truth_labels[:, class_index]

            mask_class_pred = class_pred_labels.gt(confidence_threshold)
            mask_class_labels = class_truth_labels.gt(EPSILON)

            class_pred_labels, class_pred_boxes = class_pred_labels[mask_class_pred], frame_pred_boxes[mask_class_pred]

            kept_element_indexes = nms(boxes=class_pred_boxes, scores=class_pred_labels, iou_threshold=iou_threshold)

            filtered_detection = {'boxes': torch.Tensor([]), 'scores': torch.Tensor([]), 'labels': torch.Tensor([])}

            if len(kept_element_indexes) > 0:
                filtered_detection = {"boxes": class_pred_boxes[kept_element_indexes],
                                      "scores": class_pred_labels[kept_element_indexes],
                                      "labels": torch.Tensor([int(class_index)] * len(class_pred_boxes[kept_element_indexes]))}

            filtered_pred.append(filtered_detection)

            filtered_label = {'boxes': torch.Tensor([]), 'labels': torch.Tensor([])}
            if len(mask_class_labels) > 0:
                filtered_label = {"boxes": frame_truth_boxes[mask_class_labels],
                                   "labels": torch.Tensor([int(class_index)] * len(frame_truth_boxes))}

            filtered_truth.append(filtered_label)


    return filtered_pred, filtered_truth


def mean_average_precision(pred, truth, iou_threshold=0.5):
    """
    Computes the mean average precision (mAP) for a given set of predictions and truth.
    :param truth: List of dictionaries containing the ground truths.
    :param pred: List of dictionaries containing the predictions.
    :param iou_threshold: Threshold for the IoU.
    """
    map_metric = MeanAveragePrecision(iou_thresholds=[iou_threshold], iou_type="bbox", box_format="xyxy")

    map_metric.update(pred, truth)
    values = map_metric.compute()

    return float(values['map'])


def precision_recall_f1(dataset, class_predictions, iou_threshold, label_confidence_threshold):
    """
    Computes the precision, recall and f1 score for a given set of predictions.
    :param dataset: Dataset for which the predictions were computed.
    :param class_predictions: Dictionary containing the predictions.
    :param iou_threshold: Threshold for the IoU.
    :param label_confidence_threshold: Threshold for the label confidence score.
    :return: Tuple containing the precision, recall and f1 score.
    """
    tp = 0
    fp = 0
    fn = 0

    for video_id in class_predictions:
        for frame_id in class_predictions[video_id]:
            frame_index = dataset.get_frame_index((video_id, frame_id))
            truth_labels, truth_boxes = dataset.get_labels_and_boxes(frame_index)

            matched_box_indexes = set()
            for index_i in range(len(class_predictions[video_id][frame_id])):
                if sum(class_predictions[video_id][frame_id][index_i]['labels']) < EPSILON:
                    break
                detected_box = torch.Tensor([class_predictions[video_id][frame_id][index_i]['bbox']])

                max_box_iou = 0
                max_truth_label = [0] * NUM_CLASSES
                max_truth_box = None
                for index in range(len(truth_labels)):
                    if index in matched_box_indexes:
                        continue
                    truth_box = torch.Tensor([truth_boxes[index].tolist()])
                    if truth_box.sum() == 0:
                        break

                    box_iou = single_box_iou(truth_box, detected_box)
                    if box_iou > iou_threshold:
                        if box_iou > max_box_iou:
                            max_box_iou = box_iou
                            max_truth_box = index
                            max_truth_label = truth_labels[index]

                if max_truth_box is not None:
                    matched_box_indexes.add(max_truth_box)

                detected = [1 if class_predictions[video_id][frame_id][index_i]['labels'][label] > label_confidence_threshold else 0 for label in range(NUM_CLASSES)]
                for class_index in range(NUM_CLASSES):
                    if max_truth_label[class_index] == 1 and detected[class_index] == 1:
                        tp += 1
                    elif max_truth_label[class_index] == 1 and detected[class_index] == 0:
                        fn += 1
                    elif max_truth_label[class_index] == 0 and detected[class_index] == 1:
                        fp += 1

            for index in range(len(truth_labels)):
                if index in matched_box_indexes:
                    continue
                if truth_labels[index].sum() == 0:
                    break
                for class_index in range(NUM_CLASSES):
                    if truth_labels[index][class_index] == 1:
                        fn += 1

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def count_violated_constraints(predictions, constraints, iou_threshold, label_confidence_threshold):
    """
    Counts the number of violated pairwise constraints.
    :param predictions: Dictionary containing the predictions.
    :param constraints: List of constraints.
    :param iou_threshold: Threshold for the IoU.
    :param label_confidence_threshold: Threshold for the label confidence score.
    :return: Number of violated constraints, Number of frames with a violation.
    """
    num_constraint_violations = 0
    num_frames_with_violation = 0

    constraint_violation_dict = {
        "simplex": {
            "no-agent": 0,
            "no-location": 0
        }
    }

    for video_id in predictions:
        for frame_id in predictions[video_id]:
            frame_violation = False

            for detection in predictions[video_id][frame_id]:
                detection_labels = detection['labels']

                if len(detection_labels) == 0:
                    continue

                has_agent = False
                has_location = False

                for index_i, class_i in enumerate(detection_labels):
                    if class_i < 10:
                        has_agent = True
                    if class_i == 8 or class_i == 9:
                        has_location = True
                    if class_i > 28:
                        has_location = True

                    for index_j, class_j in enumerate(detection_labels[index_i + 1:]):
                        if constraints[class_i * NUM_CLASSES + class_j][2] == 1:
                            continue

                        if class_i not in constraint_violation_dict:
                            constraint_violation_dict[class_i] = {}
                        if class_j not in constraint_violation_dict[class_i]:
                            constraint_violation_dict[class_i][class_j] = 0

                        constraint_violation_dict[class_i][class_j] += 1
                        num_constraint_violations += 1
                        frame_violation = True

                if not has_agent:
                    constraint_violation_dict["simplex"]["no-agent"] += 1
                    num_constraint_violations += 1
                    frame_violation = True
                if not has_location:
                    constraint_violation_dict["simplex"]["no-location"] += 1
                    num_constraint_violations += 1
                    frame_violation = True

            if frame_violation:
                num_frames_with_violation += 1

    return (num_constraint_violations, num_frames_with_violation, constraint_violation_dict)
