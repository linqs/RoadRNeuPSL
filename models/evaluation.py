import torch

from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms

from models.losses import single_box_iou
from utils import pixel_to_ratio_coordinates
from utils import ratio_to_pixel_coordinates
from utils import IOU_THRESHOLD
from utils import LABEL_CONFIDENCE_THRESHOLD
from utils import NUM_CLASSES


def filter_detections(frame_indexes, box_predictions, class_predictions, iou_threshold, label_confidence_threshold, num_classes=NUM_CLASSES):
    """
    Filters detections by first removing all detections with a confidence score below a threshold and then applying non-maximum suppression.
    :param frame_indexes: List of frame indexes (N, 2) for which the detections were computed.
    :param box_predictions: Tensor of shape (N, 4) containing the predicted bounding boxes.
    :param class_predictions: Tensor of shape (N, C) containing the predicted class probabilities.
    :param iou_threshold: Threshold for the IoU.
    :param label_confidence_threshold: Threshold for the confidence score.
    :param num_classes: Number of classes in the dataset.
    """
    filtered_detections = []
    filtered_detection_indexes = []

    for frame_index in range(frame_indexes.shape[0]):
        frame_boxes = box_predictions[frame_index]

        filtered_detection_indexes.append(int(frame_indexes[frame_index]))

        for class_index in range(num_classes):
            frame_scores = class_predictions[frame_index, :, class_index]

            mask_frame_scores = frame_scores.gt(label_confidence_threshold)

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
        frame_truth_labels, frame_truth_boxes = dataset.get_labels_and_boxes(frame_index)

        mask_frame_boxes = frame_truth_boxes.sum(dim=1).gt(0)

        frame_truth_boxes = frame_truth_boxes[mask_frame_boxes]
        frame_truth_labels = frame_truth_labels[mask_frame_boxes]

        for class_index in range(num_classes):
            class_frame_truth_labels = frame_truth_labels[:, class_index]

            mask_class_frame_truth_labels = class_frame_truth_labels.gt(0.5)

            if class_frame_truth_labels.sum() == 0:
                ground_truth.append({'boxes': torch.Tensor([]), 'labels': torch.Tensor([])})
                continue

            class_frame_truth_boxes = frame_truth_boxes[mask_class_frame_truth_labels]
            scaled_class_frame_truth_boxes = ratio_to_pixel_coordinates(
                class_frame_truth_boxes, dataset.image_height() / dataset.image_resize,
                dataset.image_width() / dataset.image_resize)

            class_frame_frame_labels = torch.Tensor([int(class_index)] * len(scaled_class_frame_truth_boxes)).type(torch.int64)

            ground_truth.append({'boxes': scaled_class_frame_truth_boxes, 'labels': class_frame_frame_labels})

    return ground_truth


def mean_average_precision(ground_truths, detections, iou_threshold=0.5):
    """
    Computes the mean average precision (mAP) for a given set of predictions and ground truths.
    :param ground_truths: List containing the ground truths.
    :param detections: List containing the detections.
    :param iou_threshold: Threshold for the IoU.
    """
    map_metric = MeanAveragePrecision(iou_thresholds=[iou_threshold], iou_type="bbox", box_format="xyxy")

    map_metric.update(detections, ground_truths)
    values = map_metric.compute()

    return float(values['map'])


def precision_recall_f1(dataset, predictions):
    """
    Computes the precision, recall and f1 score for a given set of predictions.
    :param dataset: Dataset for which the predictions were computed.
    :param predictions: Dictionary containing the predictions.
    :return: Tuple containing the precision, recall and f1 score.
    """
    tp = 0
    fp = 0
    fn = 0

    for video_id in predictions:
        for frame_id in predictions[video_id]:
            frame_index = dataset.get_frame_index((video_id, frame_id))
            truth_labels, truth_boxes = dataset.get_labels_and_boxes(frame_index)

            matched_box_indexes = set()
            for index in range(len(predictions[video_id][frame_id])):
                if sum(predictions[video_id][frame_id][index]['labels']) < 0.0001:
                    break
                detected_box = pixel_to_ratio_coordinates(torch.Tensor([predictions[video_id][frame_id][index]['bbox']]),
                                                          dataset.image_height() / dataset.image_resize,
                                                          dataset.image_width() / dataset.image_resize)

                max_box_iou = 0
                max_truth_label = [0] * NUM_CLASSES
                max_truth_box = None
                for index_j in range(len(truth_labels)):
                    if index_j in matched_box_indexes:
                        continue
                    truth_box = torch.Tensor([truth_boxes[index_j].tolist()])
                    if truth_box.sum() == 0:
                        break

                    box_iou = single_box_iou(truth_box, detected_box)
                    if box_iou > IOU_THRESHOLD:
                        if box_iou > max_box_iou:
                            max_box_iou = box_iou
                            max_truth_box = index_j
                            max_truth_label = truth_labels[index_j]

                if max_truth_box is not None:
                    matched_box_indexes.add(max_truth_box)

                detected = [1 if predictions[video_id][frame_id][index]['labels'][label] > LABEL_CONFIDENCE_THRESHOLD else 0 for label in range(NUM_CLASSES)]
                for class_index in range(NUM_CLASSES):
                    if max_truth_label[class_index] == 1 and detected[class_index] == 1:
                        tp += 1
                    elif max_truth_label[class_index] == 1 and detected[class_index] == 0:
                        fn += 1
                    elif max_truth_label[class_index] == 0 and detected[class_index] == 1:
                        fp += 1

            for index_j in range(len(truth_labels)):
                if index_j in matched_box_indexes:
                    continue
                if truth_labels[index_j].sum() == 0:
                    break
                for class_index in range(NUM_CLASSES):
                    if truth_labels[index_j][class_index] == 1:
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


def count_violated_constraints(predictions, constraints):
    """
    Counts the number of violated pairwise constraints.
    :param predictions: Dictionary containing the predictions.
    :param constraints: List of constraints.
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

    return num_constraint_violations, num_frames_with_violation, constraint_violation_dict
