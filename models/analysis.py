import os

import torch
import torchvision

from matplotlib import pyplot as plt

from utils import load_json_file
from utils import ratio_to_pixel_coordinates
from utils import BASE_RGB_IMAGES_DIR
from utils import BASE_TEST_RGB_IMAGES_DIR
from utils import LABEL_MAPPING
from utils import PREDICTION_LOGITS_JSON_FILENAME

BORDER_LINEWIDTH = 1
DETECTED_BORDER_COLOR = (0, 1, 0)
FONT_BOUNDING_BOX = dict(facecolor="white", alpha=0.5, edgecolor="none", pad=-0.1)
FONT_COLOR = "blue"
FONT_SIZE = 7
GROUND_TRUTH_BORDER_COLOR = (1, 0, 0)


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


def save_images_with_bounding_boxes(dataset, output_dir, write_images_with_labels, max_saved_images, labels_confidence_threshold, write_ground_truth=True, test=False):
    """
    Saves images with bounding boxes for the given dataset.
    :param dataset: Dataset for which the images should be saved.
    :param output_dir: Directory to which the images should be saved.
    :param write_images_with_labels: Whether the images should be saved with labels.
    :param max_saved_images: Maximum number of images to save.
    :param labels_confidence_threshold: Label confidence used to output labels on images with labels.
    :param write_ground_truth: Whether the images should be saved with ground truth boxes and labels.
    :param test: Whether the dataset is a test dataset.
    """
    predictions = load_json_file(os.path.join(output_dir, PREDICTION_LOGITS_JSON_FILENAME))
    label_mapping = LABEL_MAPPING

    for video_id, video_predictions in predictions.items():
        images_saved_for_video = 0
        os.makedirs(os.path.join(output_dir, "rgb-images", video_id), exist_ok=True)

        for frame_id, frame_predictions in video_predictions.items():
            if images_saved_for_video >= max_saved_images:
                break

            frame_index = dataset.get_frame_index((video_id, frame_id))
            frame_ids, pixel_values, pixel_mask, ground_truth_labels, ground_truth_boxes = dataset[frame_index]
            if test:
                load_frame_path = os.path.join(BASE_TEST_RGB_IMAGES_DIR, video_id, frame_id)
            else:
                load_frame_path = os.path.join(BASE_RGB_IMAGES_DIR, video_id, frame_id)

            if write_ground_truth:
                ground_truth_boxes = ratio_to_pixel_coordinates(ground_truth_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)

            detected_boxes = torch.Tensor([prediction["bbox"] for prediction in frame_predictions if sum(prediction["labels"]) > 0])
            detected_boxes = ratio_to_pixel_coordinates(detected_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)
            detected_labels = torch.Tensor([prediction["labels"] for prediction in frame_predictions if sum(prediction["labels"]) > 0])
            detected_labels = detected_labels.gt(labels_confidence_threshold).float()

            save_frame_path = os.path.join(output_dir, "rgb-images", video_id, frame_id[:-4] + "_boxes.jpg")
            save_frame_with_bounding_boxes(load_frame_path, save_frame_path, ground_truth_boxes, detected_boxes, ground_truth_labels, detected_labels, label_mapping, write_labels=False, write_ground_truth=write_ground_truth, write_detected=True)

            if write_images_with_labels:
                if write_ground_truth:
                    save_frame_path = os.path.join(output_dir, "rgb-images", video_id, frame_id[:-4] + "_ground_truth.jpg")
                    save_frame_with_bounding_boxes(load_frame_path, save_frame_path, ground_truth_boxes, detected_boxes, ground_truth_labels, detected_labels, label_mapping, write_labels=True, write_ground_truth=True, write_detected=False)

                save_frame_path = os.path.join(output_dir, "rgb-images", video_id, frame_id[:-4] + "_detected.jpg")
                save_frame_with_bounding_boxes(load_frame_path, save_frame_path, ground_truth_boxes, detected_boxes, ground_truth_labels, detected_labels, label_mapping, write_labels=True, write_ground_truth=False, write_detected=True)

            images_saved_for_video += 1
