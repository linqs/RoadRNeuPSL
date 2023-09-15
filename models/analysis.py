import os

import torch
import torchvision

from matplotlib import pyplot as plt

import utils
from utils import BASE_RGB_IMAGES_DIR
from utils import EVALUATION_PREDICTION_JSON_FILENAME


FONT_SIZE = 7
FONT_COLOR = "blue"
FONT_BOUNDING_BOX = dict(facecolor="white", alpha=0.5, edgecolor="none", pad=-0.1)

BORDER_LINEWIDTH = 1
GROUND_TRUTH_BORDER_COLOR = (1, 0, 0)
DETECTED_BORDER_COLOR = (0, 1, 0)


def _ratio_to_pixel_coordinates(bounding_boxes, height, width):
    if len(bounding_boxes) == 0:
        return bounding_boxes

    bounding_boxes[:, 0] *= width
    bounding_boxes[:, 1] *= height
    bounding_boxes[:, 2] *= width
    bounding_boxes[:, 3] *= height

    return bounding_boxes


def save_frame_with_bounding_boxes(load_path, save_path, ground_truth_boxes, detected_boxes, ground_truth_labels, detected_labels, label_mapping, write_labels=False, write_ground_truth=True, write_detected=True):
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
                text = [label_mapping[index] for index in range(len(ground_truth_label[:-1])) if ground_truth_label[index] == 1]
                ax.text(ground_truth_box[0], ground_truth_box[3], "\n".join(text), fontsize=FONT_SIZE, color=FONT_COLOR, bbox=FONT_BOUNDING_BOX)

    if write_detected:
        for detected_box, detected_label in zip(detected_boxes, detected_labels):
            rectangle = plt.Rectangle((detected_box[0], detected_box[1]), detected_box[2] - detected_box[0], detected_box[3] - detected_box[1], linewidth=BORDER_LINEWIDTH, edgecolor=DETECTED_BORDER_COLOR, facecolor="none")
            ax.add_patch(rectangle)

            if write_labels:
                text = [label_mapping[index] for index in range(len(detected_label)) if detected_label[index] == 1]
                ax.text(detected_box[0], detected_box[3], "\n".join(text), fontsize=FONT_SIZE, color=FONT_COLOR, bbox=FONT_BOUNDING_BOX)

    plt.savefig(save_path)
    plt.close()


def save_images_with_bounding_boxes(dataset, arguments, max_images_saved_per_video=10, labels_threshold=0.2):
    """
    Saves images with bounding boxes for the given dataset.
    :param dataset: Dataset for which the images should be saved.
    :param arguments: Arguments passed to the program.
    :param box_confidence_threshold: Confidence threshold for bounding boxes.
    :param max_images_saved_per_video: Maximum number of images saved per video.
    """
    predictions = utils.load_json_file(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_JSON_FILENAME))
    label_mapping = dataset.label_mapping()

    for video_id, video_predictions in predictions.items():
        images_saved_for_video = 0
        os.makedirs(os.path.join(arguments.output_dir, "rgb-images", video_id), exist_ok=True)

        for frame_id, frame_predictions in video_predictions.items():
            if images_saved_for_video >= max_images_saved_per_video:
                break

            frame_index = dataset.video_id_frame_id_to_frame_index[(video_id, frame_id)]
            load_frame_path = os.path.join(BASE_RGB_IMAGES_DIR, video_id, frame_id)

            ground_truth_bounding_boxes = dataset[frame_index][3]
            prediction_bounding_boxes = torch.Tensor([prediction["bbox"] for prediction in frame_predictions if sum(prediction["labels"]) > 0])
            ground_truth_bounding_boxes = _ratio_to_pixel_coordinates(ground_truth_bounding_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)
            prediction_bounding_boxes = _ratio_to_pixel_coordinates(prediction_bounding_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)

            ground_truth_labels = dataset[frame_index][2]
            prediction_labels = torch.Tensor([prediction["labels"] for prediction in frame_predictions if sum(prediction["labels"]) > 0])
            prediction_labels = prediction_labels.gt(labels_threshold).float()

            save_frame_path = os.path.join(arguments.output_dir, "rgb-images", video_id, frame_id[:-4] + "_boxes.jpg")
            save_frame_with_bounding_boxes(load_frame_path, save_frame_path, ground_truth_bounding_boxes, prediction_bounding_boxes, ground_truth_labels, prediction_labels, label_mapping, write_labels=False, write_ground_truth=True, write_detected=True)

            save_frame_path = os.path.join(arguments.output_dir, "rgb-images", video_id, frame_id[:-4] + "_ground_truth.jpg")
            save_frame_with_bounding_boxes(load_frame_path, save_frame_path, ground_truth_bounding_boxes, prediction_bounding_boxes, ground_truth_labels, prediction_labels, label_mapping, write_labels=True, write_ground_truth=True, write_detected=False)

            save_frame_path = os.path.join(arguments.output_dir, "rgb-images", video_id, frame_id[:-4] + "_detected.jpg")
            save_frame_with_bounding_boxes(load_frame_path, save_frame_path, ground_truth_bounding_boxes, prediction_bounding_boxes, ground_truth_labels, prediction_labels, label_mapping, write_labels=True, write_ground_truth=False, write_detected=True)

            images_saved_for_video += 1
