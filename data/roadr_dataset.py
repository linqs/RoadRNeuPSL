import logging
import os
import sys

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from transformers import DetrImageProcessor

import utils

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import load_json_file
from utils import BASE_RGB_IMAGES_DIR
from utils import BASE_TEST_RGB_IMAGES_DIR
from utils import IMAGE_HEIGHT
from utils import IMAGE_WIDTH
from utils import LABEL_TYPES
from utils import NUM_CLASSES
from utils import NUM_QUERIES
from utils import ORIGINAL_LABEL_MAPPING

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


class RoadRDataset(Dataset):
    def __init__(self,
                 videos,
                 annotations_path,
                 image_resize,
                 max_frames=0,
                 test=False,
                 use_transforms=False):
        self.videos = videos
        self.annotations_path = annotations_path
        self.test = test
        self.use_transforms = use_transforms

        self.base_rgb_images_dir = BASE_TEST_RGB_IMAGES_DIR if self.test else BASE_RGB_IMAGES_DIR

        self.image_resize = image_resize
        self.max_frames = max_frames

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size={"shortest_edge": self.image_height(), "longest_edge": self.image_width()})

        self.transforms = v2.Compose([
            v2.RandomIoUCrop(),
            v2.Resize(size=(self.image_height(), self.image_width()), antialias=True),
            v2.SanitizeBoundingBoxes()
        ])

        self.annotations = None
        if not self.test:
            self.annotations = load_json_file(self.annotations_path)["db"]

        logging.info("Loaded annotations from {0}".format(self.annotations_path))

        self.frame_ids = []
        self.frame_indexes = {}
        self.load_labels = False
        self.load_frame_ids()
        logging.info("Total frames counted in all videos: {0}".format(len(self.frame_ids)))

    def set_use_transforms(self, use_transforms):
        self.use_transforms = use_transforms

    def load_frame_ids(self):
        for videoname in self.videos:
            num_video_frames = 0

            rgb_images_dir = os.path.join(self.base_rgb_images_dir, videoname)

            for frame_file_name in sorted(os.listdir(rgb_images_dir)):
                if self.max_frames != 0 and num_video_frames >= self.max_frames:
                    break

                if (not self.test) and ("annos" not in self.annotations[videoname]['frames'][str(int(frame_file_name.strip(".jpg")))]):
                    continue
                else:
                    if not self.test:
                        assert int(frame_file_name.strip(".jpg")) == self.annotations[videoname]['frames'][str(int(frame_file_name.strip(".jpg")))]['rgb_image_id']

                self.frame_indexes[(videoname, frame_file_name)] = len(self.frame_ids)
                self.frame_ids.append([videoname, frame_file_name])
                num_video_frames += 1

    def load_frame(self, frame_index, load_image=True):
        videoname, framename = self.frame_ids[frame_index]

        image = {'pixel_values': [[]], 'pixel_mask': [[]]}
        if load_image:
            image = self.processor(Image.open(os.path.join(self.base_rgb_images_dir, videoname, framename)))

        if self.test:
            return image['pixel_values'][0], image['pixel_mask'][0], torch.tensor([]), torch.tensor([])

        annotations = self.annotations[videoname]['frames'][str(int(framename[:-4]))]

        frame_labels = torch.zeros(size=(NUM_QUERIES, NUM_CLASSES + 1), dtype=torch.int8)
        frame_boxes = torch.zeros(size=(NUM_QUERIES, 4), dtype=torch.float32)

        for bounding_box_index, bounding_box in enumerate(annotations['annos']):
            frame_boxes[bounding_box_index] = torch.Tensor(annotations['annos'][bounding_box]['box'])
            frame_labels[bounding_box_index, -1] = 1  # Set the last class to 1 to indicate that there is an object box here.

            for label_type in LABEL_TYPES:
                for label_id in annotations['annos'][bounding_box][label_type + '_ids']:
                    if int(label_id) not in ORIGINAL_LABEL_MAPPING[label_type]:
                        continue
                    frame_labels[bounding_box_index][ORIGINAL_LABEL_MAPPING[label_type][int(label_id)][0]] = 1

        img = image['pixel_values'][0]
        mask = image['pixel_mask'][0]
        if self.use_transforms:
            img, mask, frame_labels, frame_boxes = self.transform_frame(image, frame_labels, frame_boxes)

        return img, mask, frame_labels, frame_boxes

    def transform_frame(self, image, frame_labels, frame_boxes):
        img = image['pixel_values'][0]
        mask = image['pixel_mask'][0]
        if self.use_transforms:
            scaled_frame_boxes = utils.ratio_to_pixel_coordinates(frame_boxes, self.image_height(), self.image_width())
            tv_frame_boxes = tv_tensors.BoundingBoxes(scaled_frame_boxes, format="XYXY", canvas_size=(self.image_height(), self.image_width()))
            target = {"boxes": tv_frame_boxes, "labels": frame_labels, "masks": image['pixel_mask'][0]}
            img = tv_tensors.Image(img)

            transformed_img, transformed_target = self.transforms(img, target)

            frame_labels = torch.zeros(size=(NUM_QUERIES, NUM_CLASSES + 1), dtype=torch.int8)
            frame_boxes = torch.zeros(size=(NUM_QUERIES, 4), dtype=torch.float32)

            for bounding_box_index, bounding_box in enumerate(transformed_target["boxes"]):
                frame_boxes[bounding_box_index] = transformed_target["boxes"][bounding_box_index]
                frame_labels[bounding_box_index] = transformed_target["labels"][bounding_box_index]

            frame_boxes = utils.pixel_to_ratio_coordinates(frame_boxes, self.image_height(), self.image_width())

            mask = transformed_target["masks"]
            img = transformed_img

        return img, mask, frame_labels, frame_boxes

    def image_height(self):
        return int(IMAGE_HEIGHT * self.image_resize)

    def image_width(self):
        return int(IMAGE_WIDTH * self.image_resize)

    def get_frame_id(self, frame_index):
        return self.frame_ids[frame_index]

    def get_frame_index(self, frame_id):
        return self.frame_indexes[(frame_id[0], frame_id[1])]

    def get_labels_and_boxes(self, frame_index):
        _, _, labels, boxes = self.load_frame(frame_index, load_image=False)
        return labels, boxes

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, frame_index):
        pixel_values, pixel_mask, labels, boxes = self.load_frame(frame_index)
        return frame_index, pixel_values, pixel_mask, labels, boxes
