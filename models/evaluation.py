import os

import numpy as np
import torch

import torchvision

from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms
from matplotlib import pyplot as plt

from models.losses import single_box_iou
from utils import ratio_to_pixel_coordinates
from utils import AGENT_CLASSES
from utils import BASE_RGB_IMAGES_DIR
from utils import BASE_TEST_RGB_IMAGES_DIR
from utils import LABEL_MAPPING
from utils import NUM_CLASSES

EPSILON = 1e-7

BORDER_LINEWIDTH = 1
DETECTED_BORDER_COLOR = (0, 1, 0)
FONT_BOUNDING_BOX = dict(facecolor="white", alpha=0.5, edgecolor="none", pad=-0.1)
FONT_COLOR = "blue"
FONT_SIZE = 7
GROUND_TRUTH_BORDER_COLOR = (1, 0, 0)


def sort_detections_in_frames(pred_labels, pred_boxes):
    """
    Sorts the detections in each frame by their confidence score.
    :param pred_labels: List of lists of predicted labels.
    :param pred_boxes: List of lists of predicted boxes.
    :return: Sorted lists of predicted labels and boxes.
    """
    sorted_boxes = []
    sorted_labels = []

    for frame_pred_labels, frame_pred_boxes in zip(pred_labels, pred_boxes):
        frame_pred_labels, frame_pred_boxes = zip(*sorted(zip(frame_pred_labels, frame_pred_boxes), key=lambda x: x[0][-1], reverse=True))

        sorted_boxes.append(frame_pred_boxes)
        sorted_labels.append(frame_pred_labels)

    return sorted_boxes, sorted_labels


def agent_nms_mask(dataset, frame_ids, box_predictions, class_predictions, iou_threshold, label_confidence_threshold, box_confidence_threshold):
    """
    Applies non-maximum suppression to the detections using the agent class.
    :param dataset: Dataset for which the detections were computed.
    :param frame_ids: List of frame ids (video name, frame name).
    :param box_predictions: The predicted bounding boxes.
    :param class_predictions: The predicted class labels in one-hot encoding.
    :param iou_threshold: Threshold for the IoU.
    :param label_confidence_threshold: Threshold for the label confidence score.
    :param box_confidence_threshold: Threshold for the box confidence score.
    :return: Integer mask of the kept detections.
    """
    mask_keep_detections = []
    for frame_id in frame_ids:
        frame_index = dataset.get_frame_index(frame_id)

        mask_keep_detections.append([0] * len(box_predictions[frame_index]))

        frame_pred_labels, frame_pred_boxes = class_predictions[frame_index], box_predictions[frame_index]

        mask_frame_pred = frame_pred_labels[:, -1].gt(box_confidence_threshold)

        frame_pred_labels, frame_pred_boxes = frame_pred_labels[mask_frame_pred], frame_pred_boxes[mask_frame_pred]

        for class_index in AGENT_CLASSES:
            class_pred_labels = frame_pred_labels[:, class_index]

            mask_class_pred = class_pred_labels.gt(label_confidence_threshold)

            class_pred_labels, class_pred_boxes = class_pred_labels[mask_class_pred], frame_pred_boxes[mask_class_pred]

            nms_keep_indexes = nms(boxes=class_pred_boxes, scores=class_pred_labels, iou_threshold=iou_threshold)

            kept_indices = np.arange(len(box_predictions[frame_index]))[mask_frame_pred][mask_class_pred][nms_keep_indexes].tolist()
            kept_indices = [kept_indices] if isinstance(kept_indices, int) else kept_indices
            for index in kept_indices:
                if mask_keep_detections[-1][index] == 0:
                    mask_keep_detections[-1][index] = 1

    return mask_keep_detections


def filter_map_pred_and_truth(dataset, frame_ids, box_predictions, class_predictions, iou_threshold, label_confidence_threshold, box_confidence_threshold):
    """
    Filters predictions using non-maximum suppression and returns the predictions and truth in the format
    required for the mean average precision metric.
    :param dataset: Dataset for which the detections were computed.
    :param frame_ids: List of frame ids (video name, frame name).
    :param box_predictions: Tensor of shape (N, 4) containing the predicted bounding boxes.
    :param class_predictions: Tensor of shape (N, C) containing the predicted class probabilities.
    :param iou_threshold: Threshold for the IoU.
    :param label_confidence_threshold: Threshold for the label confidence score.
    :param box_confidence_threshold: Threshold for the box confidence score.
    :return: Filtered predictions and truth.
    """
    filtered_pred = []
    filtered_truth = []

    for frame_id in frame_ids:
        frame_index = dataset.get_frame_index(frame_id)

        frame_pred_labels, frame_pred_boxes = class_predictions[frame_index], box_predictions[frame_index]
        frame_truth_labels, frame_truth_boxes = dataset.get_labels_and_boxes(frame_index)

        mask_frame_pred = frame_pred_labels[:, -1].gt(box_confidence_threshold)
        mask_frame_truth = frame_truth_boxes.sum(dim=1).gt(EPSILON)

        frame_pred_labels, frame_pred_boxes = frame_pred_labels[mask_frame_pred], frame_pred_boxes[mask_frame_pred]
        frame_truth_labels, frame_truth_boxes = frame_truth_labels[mask_frame_truth], frame_truth_boxes[mask_frame_truth]

        frame_pred_boxes = ratio_to_pixel_coordinates(frame_pred_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)
        frame_truth_boxes = ratio_to_pixel_coordinates(frame_truth_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)

        for class_index in range(NUM_CLASSES):
            class_pred_labels = frame_pred_labels[:, class_index]
            class_truth_labels = frame_truth_labels[:, class_index]

            mask_class_pred = class_pred_labels.gt(label_confidence_threshold)
            mask_class_labels = class_truth_labels.gt(EPSILON)

            class_pred_labels, class_pred_boxes = class_pred_labels[mask_class_pred], frame_pred_boxes[mask_class_pred]

            kept_element_indexes = nms(boxes=class_pred_boxes, scores=class_pred_labels, iou_threshold=iou_threshold)

            filtered_detection = {'boxes': torch.Tensor([]), 'scores': torch.Tensor([]), 'labels': torch.Tensor([])}

            if len(kept_element_indexes) > 0:
                filtered_detection = {"boxes": class_pred_boxes[kept_element_indexes],
                                      "scores": class_pred_labels[kept_element_indexes],
                                      "labels": torch.Tensor([int(class_index)] * len(class_pred_boxes[kept_element_indexes])).int()}

            filtered_pred.append(filtered_detection)

            filtered_label = {'boxes': torch.Tensor([]), 'labels': torch.Tensor([])}
            if len(mask_class_labels) > 0:
                filtered_label = {"boxes": frame_truth_boxes[mask_class_labels],
                                   "labels": torch.Tensor([int(class_index)] * len(frame_truth_boxes[mask_class_labels])).int()}

            filtered_truth.append(filtered_label)

    return filtered_pred, filtered_truth


def mean_average_precision(dataset, frame_ids, box_predictions, class_predictions, iou_threshold, label_confidence_threshold, box_confidence_threshold):
    """
    Computes the mean average precision (mAP) for a given set of predictions and truth.
    :param dataset: Dataset for which the detections were computed.
    :param frame_ids: List of frame ids (video name, frame name).
    :param box_predictions: Tensor of shape (N, 4) containing the predicted bounding boxes.
    :param class_predictions: Tensor of shape (N, C) containing the predicted class probabilities.
    :param iou_threshold: Threshold for the IoU.
    :param label_confidence_threshold: Threshold for the label confidence score.
    :param box_confidence_threshold: Threshold for the box confidence score.
    :param iou_threshold: Threshold for the IoU.
    """
    pred, truth = filter_map_pred_and_truth(dataset, frame_ids, box_predictions, class_predictions, iou_threshold, label_confidence_threshold, box_confidence_threshold)

    map_metric = MeanAveragePrecision(iou_thresholds=[iou_threshold], iou_type="bbox", box_format="xyxy")

    map_metric.update(pred, truth)
    values = map_metric.compute()

    return float(values['map'])


def match_box(pred_box, truth_boxes, skip_box_indexes, iou_threshold):
    """
    Matches a predicted box to a ground truth box.
    :param pred_box: A single box of shape (1, 4).
    :param truth_boxes: A tensor of shape (N, 4) containing the ground truth boxes.
    :param skip_box_indexes: A set of indexes of boxes that have already been matched.
    :param iou_threshold: Threshold for the IoU.
    :return: The index of the matched ground truth box or None if no match was found.
    """
    max_box_iou = 0
    max_truth_box_index = None
    for truth_box_index in range(len(truth_boxes)):
        if truth_box_index in skip_box_indexes:
            continue

        truth_box = torch.Tensor([truth_boxes[truth_box_index].tolist()])
        if truth_box.sum() < EPSILON:
            break

        box_iou = single_box_iou(truth_box, pred_box)
        if box_iou > iou_threshold:
            if box_iou > max_box_iou:
                max_box_iou = box_iou
                max_truth_box_index = truth_box_index

    return max_truth_box_index


def precision_recall_f1(dataset, frame_ids, class_predictions, box_predictions, iou_threshold, label_confidence_threshold, box_confidence_threshold):
    """
    Computes the precision, recall and f1 score for a given set of predictions.
    :param dataset: Dataset for which the predictions were computed.
    :param frame_ids: List of frame ids (video name, frame name).
    :param class_predictions: List containing the predictions.
    :param iou_threshold: Threshold for the IoU.
    :param label_confidence_threshold: Threshold for the label confidence score.
    :param box_confidence_threshold: Threshold for the box confidence score.
    :return: Tuple containing the precision, recall and f1 score.
    """
    tp = 0
    fp = 0
    fn = 0

    for frame_id in frame_ids:
        frame_index = dataset.get_frame_index(frame_id)

        truth_labels, truth_boxes = dataset.get_labels_and_boxes(frame_index)

        matched_box_indexes = set()
        for pred_box_index in range(len(class_predictions[frame_index]) - 1):
            if class_predictions[frame_index][pred_box_index][-1] < box_confidence_threshold:
                break
            detected_box = torch.Tensor([box_predictions[frame_index][pred_box_index]])

            truth_box_index = match_box(detected_box, truth_boxes, matched_box_indexes, iou_threshold)

            truth_label = [0] * NUM_CLASSES

            if truth_box_index is not None:
                matched_box_indexes.add(truth_box_index)
                truth_label = truth_labels[truth_box_index]

            detected_label = [1 if class_predictions[frame_index][pred_box_index][label] > label_confidence_threshold else 0 for label in range(NUM_CLASSES)]
            for class_index in range(NUM_CLASSES):
                if truth_label[class_index] == 1 and detected_label[class_index] == 1:
                    tp += 1
                elif truth_label[class_index] == 1 and detected_label[class_index] == 0:
                    fn += 1
                elif truth_label[class_index] == 0 and detected_label[class_index] == 1:
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


def count_violated_constraints(constraints, dataset, frame_ids, class_predictions, label_confidence_threshold, box_confidence_threshold):
    """
    Counts the number of violated pairwise constraints.
    :param constraints: List of pairwise constraints.
    :param dataset: Dataset for which the predictions were computed.
    :param frame_ids: List of frame ids (video name, frame name).
    :param class_predictions: List containing the predictions.
    :param label_confidence_threshold: Threshold for the label confidence score.
    :param box_confidence_threshold: Threshold for the box confidence score.
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

    for frame_id in frame_ids:
        frame_index = dataset.get_frame_index(frame_id)

        frame_violation = False

        for pred_box_index in range(len(class_predictions[frame_index]) - 1):
            if class_predictions[frame_index][pred_box_index][-1] < box_confidence_threshold:
                break

            detected_label = [1 if class_predictions[frame_index][pred_box_index][label] > label_confidence_threshold else 0 for label in range(NUM_CLASSES)]

            has_agent = False
            has_location = False

            for index_i, class_i in enumerate(detected_label):
                if class_i < 10:
                    has_agent = True
                if class_i == 8 or class_i == 9:
                    has_location = True
                if class_i > 28:
                    has_location = True

                for index_j, class_j in enumerate(detected_label[index_i + 1:]):
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


def save_frame_with_bounding_boxes(load_path, save_path, ground_truth_boxes, detected_boxes, ground_truth_labels, detected_labels, write_labels=False, write_ground_truth=True, write_detected=True):
    """
    Saves the given frame with the given bounding boxes.
    :param load_path: Path from which the frame should be loaded.
    :param save_path: Path to which the frame should be saved.
    :param ground_truth_boxes: List of ground truth bounding boxes.
    :param detected_boxes: List of detected bounding boxes.
    """
    frame = torchvision.io.read_image(load_path)

    fig, ax = plt.subplots()
    ax.imshow(frame.permute(1, 2, 0))

    if write_ground_truth:
        for ground_truth_box, ground_truth_label in zip(ground_truth_boxes, ground_truth_labels):
            rectangle = plt.Rectangle((ground_truth_box[0], ground_truth_box[1]), ground_truth_box[2] - ground_truth_box[0], ground_truth_box[3] - ground_truth_box[1], linewidth=BORDER_LINEWIDTH, edgecolor=GROUND_TRUTH_BORDER_COLOR, facecolor="none")
            ax.add_patch(rectangle)

            if write_labels:
                text = [LABEL_MAPPING[index] for index in range(len(ground_truth_label[:-1])) if ground_truth_label[index] == 1]
                ax.text(ground_truth_box[0], ground_truth_box[3], "\n".join(text), fontsize=FONT_SIZE, color=FONT_COLOR, bbox=FONT_BOUNDING_BOX)

    if write_detected:
        for detected_box, detected_label in zip(detected_boxes, detected_labels):
            rectangle = plt.Rectangle((detected_box[0], detected_box[1]), detected_box[2] - detected_box[0], detected_box[3] - detected_box[1], linewidth=BORDER_LINEWIDTH, edgecolor=DETECTED_BORDER_COLOR, facecolor="none")
            ax.add_patch(rectangle)

            if write_labels:
                text = [LABEL_MAPPING[index] for index in range(len(detected_label)) if detected_label[index] == 1]
                ax.text(detected_box[0], detected_box[3], "\n".join(text), fontsize=FONT_SIZE, color=FONT_COLOR, bbox=FONT_BOUNDING_BOX)

    plt.savefig(save_path)
    plt.close()


def save_images_with_bounding_boxes(output_dir, dataset, frame_ids, class_predictions, box_predictions, labels_confidence_threshold, box_confidence_threshold, max_saved_images=10, test=False):
    """
    Saves images with bounding boxes for the given dataset.
    :param dataset: Dataset for which the predictions were computed.
    :param frame_ids: List of frame ids (video name, frame name).
    :param class_predictions: List containing the predictions.
    :param output_dir: Directory to which the images should be saved.
    :param max_saved_images: Maximum number of images to save.
    :param labels_confidence_threshold: Label confidence used to output labels on images with labels.
    :param box_confidence_threshold: Label confidence used to output labels on images with labels.
    :param test: Whether the dataset is a test dataset.
    """

    saved_image_dict = {}

    for frame_id in frame_ids:
        if frame_id[0] not in saved_image_dict:
            saved_image_dict[frame_id[0]] = 0
            os.makedirs(os.path.join(output_dir, "rgb-images", frame_id[0]), exist_ok=True)

        if saved_image_dict[frame_id[0]] >= max_saved_images:
            continue

        frame_index = dataset.get_frame_index(frame_id)
        ground_truth_labels, ground_truth_boxes = dataset.get_labels_and_boxes(frame_index)

        load_frame_path = os.path.join(BASE_TEST_RGB_IMAGES_DIR, frame_id[0], frame_id[1])

        if not test:
            load_frame_path = os.path.join(BASE_RGB_IMAGES_DIR, frame_id[0], frame_id[1])
            ground_truth_boxes = ratio_to_pixel_coordinates(ground_truth_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)

        detected_boxes = torch.Tensor([box_prediction for box_prediction, class_prediction in zip(box_predictions[frame_index], class_predictions[frame_index]) if class_prediction[-1] > box_confidence_threshold])
        detected_labels = torch.Tensor([class_prediction[:-1] for class_prediction in class_predictions[frame_index] if class_prediction[-1] > box_confidence_threshold])
        detected_labels = detected_labels.gt(labels_confidence_threshold).float()

        save_frame_path = os.path.join(output_dir, "rgb-images", frame_id[0], frame_id[1][:-4] + "_boxes.jpg")
        save_frame_with_bounding_boxes(load_frame_path, save_frame_path, ground_truth_boxes, detected_boxes, ground_truth_labels, detected_labels, write_labels=False, write_ground_truth=(not test), write_detected=True)

        if not test:
            save_frame_path = os.path.join(output_dir, "rgb-images", frame_id[0], frame_id[1][:-4] + "_ground_truth.jpg")
            save_frame_with_bounding_boxes(load_frame_path, save_frame_path, ground_truth_boxes, detected_boxes, ground_truth_labels, detected_labels, write_labels=True, write_ground_truth=True, write_detected=False)

        save_frame_path = os.path.join(output_dir, "rgb-images", frame_id[0], frame_id[1][:-4] + "_detected.jpg")
        save_frame_with_bounding_boxes(load_frame_path, save_frame_path, ground_truth_boxes, detected_boxes, ground_truth_labels, detected_labels, write_labels=True, write_ground_truth=False, write_detected=True)

        saved_image_dict[frame_id[0]] += 1
