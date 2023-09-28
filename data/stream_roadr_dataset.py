import logging
import os
import sys
import time

import torch
from PIL.Image import Image
from torch.utils.data import Dataset
from transformers import DetrImageProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger

from utils import load_json_file
from utils import BASE_RGB_IMAGES_DIR
from utils import BASE_RGB_IMAGES_PROCESSED_DIR
from utils import LABEL_TYPES
from utils import NUM_CLASSES
from utils import ORIGINAL_LABEL_MAPPING
from utils import VIDEO_PARTITIONS

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

IMAGE_HEIGHT = 960
IMAGE_WIDTH = 1280


class RoadRDataset(Dataset):
    def __init__(self,
                 videos,
                 data_path,
                 image_resize,
                 num_queries=100,
                 max_frames=0):
        self.videos = videos
        self.data_path = data_path
        self.image_resize = image_resize
        self.num_queries = num_queries
        self.max_frames = max_frames

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size={"shortest_edge": self.image_height(), "longest_edge": self.image_width()})

        self.frame_ids = []

        self.num_workers = os.cpu_count()
        logging.info("Number of workers used for data loading: {0}".format(self.num_workers))

        self.database = load_json_file(self.data_path)['db']
        logging.info("Loaded database from {0}".format(self.data_path))

        self.load_frame_ids()
        logging.info("Total frames counted in all videos: {0}".format(len(self.frame_ids)))

        sys.exit(0)

    def load_frame_ids(self):
        for videoname in sorted(self.database.keys()):
            if not videoname in self.videos:
                continue

            num_video_frames = 0

            for frame_name in sorted([int(frame_name) for frame_name in self.database[videoname]['frames'].keys()]):
                if self.max_frames != 0 and num_video_frames >= self.max_frames:
                    break

                if "annos" not in self.database[videoname]['frames'][str(frame_name)]:
                    continue

                self.frame_ids.append([videoname, frame_name['rgb_image_id']])
                num_video_frames += 1

    def load_frame(self, videoname, framename):
        if os.path.isfile(BASE_RGB_IMAGES_PROCESSED_DIR, videoname, "{0:05d}.json".format(framename)):
            frame
        image = self.processor(Image.open(os.path.join(BASE_RGB_IMAGES_DIR, videoname, "{0:05d}.jpg".format(framename))))

        frame = self.database[videoname]['frames'][str(framename)]

        frame_labels = torch.zeros(size=(self.num_queries, NUM_CLASSES + 1), dtype=torch.int8)
        frame_boxes = torch.zeros(size=(self.num_queries, 4), dtype=torch.float32)

        for bounding_box_index, bounding_box in enumerate(frame['annos']):
            frame_boxes[bounding_box_index] = frame['annos'][bounding_box]['box']
            frame_labels[bounding_box_index, -1] = 1  # Set the last class to 1 to indicate that there is an object box here.

            for label_type in LABEL_TYPES:
                for label_id in frame['annos'][bounding_box][label_type + '_ids']:
                    if int(label_id) not in ORIGINAL_LABEL_MAPPING[label_type]:
                        continue
                    frame_labels[bounding_box_index][ORIGINAL_LABEL_MAPPING[label_type][int(label_id)][0]] = 1

        return image['pixel_values'][0], image['pixel_mask'][0], frame_labels, frame_boxes

    def image_height(self):
        return int(IMAGE_HEIGHT * self.image_resize)

    def image_width(self):
        return int(IMAGE_WIDTH * self.image_resize)

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):
        start_time = time.time()
        pixel_values, pixel_mask, labels, boxes = self.load_frame(self.frame_ids[index])
        logging.debug("Loaded frame {0} in {1:.2f} seconds.".format(self.frame_ids[index], time.time() - start_time))
        return self.frame_ids[index], pixel_values, pixel_mask, labels, boxes


if __name__ == "__main__":
    logger.initLogging("DEBUG")
    dataset = RoadRDataset(VIDEO_PARTITIONS["task1"]["TRAIN"], os.path.join(THIS_DIR, "road_trainval_v1.0.json"), 1.0, max_frames=10)