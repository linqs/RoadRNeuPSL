import os

import torch
import torchvision

from matplotlib import pyplot as plt

import utils
from utils import BASE_RGB_IMAGES_DIR
from utils import EVALUATION_PREDICTION_JSON_FILENAME


def _ratio_to_pixel_coordinates(bounding_boxes, height, width):
    if len(bounding_boxes) == 0:
        return bounding_boxes

    bounding_boxes[:, 0] *= width
    bounding_boxes[:, 1] *= height
    bounding_boxes[:, 2] *= width
    bounding_boxes[:, 3] *= height

    return bounding_boxes


def save_frame_with_bounding_boxes(load_path, save_path, ground_truth_boxes, detected_boxes):
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

    for ground_truth_box in ground_truth_boxes:
        rectangle = plt.Rectangle((ground_truth_box[0], ground_truth_box[1]), ground_truth_box[2] - ground_truth_box[0], ground_truth_box[3] - ground_truth_box[1], linewidth=1, edgecolor=(1, 0, 0), facecolor="none")
        ax.add_patch(rectangle)

    for detected_box in detected_boxes:
        rectangle = plt.Rectangle((detected_box[0], detected_box[1]), detected_box[2] - detected_box[0], detected_box[3] - detected_box[1], linewidth=1, edgecolor=(0, 1, 0), facecolor="none")
        ax.add_patch(rectangle)

    plt.savefig(save_path)
    plt.close()


def save_images_with_bounding_boxes(dataset, arguments, max_images_saved_per_video=10):
    """
    Saves images with bounding boxes for the given dataset.
    :param dataset: Dataset for which the images should be saved.
    :param arguments: Arguments passed to the program.
    :param box_confidence_threshold: Confidence threshold for bounding boxes.
    :param max_images_saved_per_video: Maximum number of images saved per video.
    """
    predictions = utils.load_json_file(os.path.join(arguments.output_dir, EVALUATION_PREDICTION_JSON_FILENAME))

    for video_id, video_predictions in predictions.items():
        images_saved_for_video = 0
        for frame_id, frame_predictions in video_predictions.items():
            if images_saved_for_video >= max_images_saved_per_video:
                break

            frame_index = dataset.video_id_frame_id_to_frame_index[(video_id, frame_id)]

            ground_truth_bounding_boxes = dataset[frame_index][3]
            prediction_bounding_boxes = torch.Tensor([prediction["bbox"] for prediction in frame_predictions if sum(prediction["labels"]) > 0])

            ground_truth_bounding_boxes = _ratio_to_pixel_coordinates(ground_truth_bounding_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)
            prediction_bounding_boxes = _ratio_to_pixel_coordinates(prediction_bounding_boxes, dataset.image_height() / dataset.image_resize, dataset.image_width() / dataset.image_resize)

            load_frame_path = os.path.join(BASE_RGB_IMAGES_DIR, video_id, frame_id)
            save_frame_path = os.path.join(arguments.output_dir, "rgb-images", video_id, frame_id)

            os.makedirs(os.path.dirname(save_frame_path), exist_ok=True)

            save_frame_with_bounding_boxes(load_frame_path, save_frame_path, ground_truth_bounding_boxes, prediction_bounding_boxes)
            images_saved_for_video += 1
