import json
import logging
import os

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
from transformers import DetrImageProcessor

from utils import NUM_CLASSES
from utils import ORIGINAL_LABEL_MAPPING

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

IMAGE_HEIGHT = 960
IMAGE_WIDTH = 1280

LABEL_TYPES = ['agent', 'action', 'loc']


class RoadRDataset(Dataset):
    def __init__(self,
                 videos,
                 data_path,
                 image_resize,
                 num_queries=100,
                 start_frame_percentage=0.0,
                 end_frame_percentage=1.0,
                 max_frames=0):
        self.labeled_videos = videos
        self.data_path = data_path
        self.image_resize = image_resize
        self.num_queries = num_queries
        self.max_frames = max_frames
        self.start_frame_percentage = start_frame_percentage
        self.end_frame_percentage = end_frame_percentage
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size={"shortest_edge": self.image_height(), "longest_edge": self.image_width()})

        self.frames = None
        self.frame_ids = None
        self.pixel_values = None
        self.pixel_mask = None
        self.labels = None
        self.boxes = None

        self.video_id_frame_id_to_frame_index = {}

        self.load_data()

    def load_data(self):
        logging.info("Loading raw json data: {0}".format(self.data_path))
        with open(self.data_path, 'r') as data_file:
            json_data = json.load(data_file)

        database = json_data['db']

        logging.debug("Counting total frames in all videos for allocation.")
        num_frames = 0
        for videoname in sorted(database.keys()):
            if not videoname in self.labeled_videos:
                continue

            num_video_frames = 0

            start_frame = int(len(database[videoname]['frames']) * self.start_frame_percentage)
            end_frame = int(len(database[videoname]['frames']) * self.end_frame_percentage)

            logging.debug("Counting valid frames for video: {0} Start frame: {1} End frame: {2}".format(videoname, start_frame, end_frame))

            for frame_name in sorted([int(frame_name) for frame_name in database[videoname]['frames'].keys()])[start_frame:end_frame]:
                if self.max_frames != 0 and num_video_frames >= self.max_frames:
                    break

                frame = database[videoname]['frames'][str(frame_name)]

                if "annos" not in frame:
                    continue

                num_frames += 1
                num_video_frames += 1
        logging.debug("Total frames counted in all videos: {0}".format(num_frames))

        self.frames = np.empty(shape=(num_frames, 2), dtype=object)  # (video_id, frame_id)

        self.frame_ids = torch.arange(num_frames, dtype=torch.int64)
        self.pixel_values = np.empty(shape=(num_frames, 3, self.image_height(), self.image_width()), dtype=np.float32)
        self.pixel_mask = np.empty(shape=(num_frames, self.image_height(), self.image_width()), dtype=np.float32)
        self.labels = torch.empty(size=(num_frames, self.num_queries, NUM_CLASSES + 1), dtype=torch.float32)
        self.boxes = torch.empty(size=(num_frames, self.num_queries, 4), dtype=torch.float32)

        logging.info("Loading frames for all videos.")
        frame_index = 0
        for videoname in sorted(database.keys()):
            if not videoname in self.labeled_videos:
                continue

            num_video_frames = 0

            start_frame = int(len(database[videoname]['frames']) * self.start_frame_percentage)
            end_frame = int(len(database[videoname]['frames']) * self.end_frame_percentage)
            logging.debug("Loading valid frames for video: {0} Start frame: {1} End frame: {2}".format(videoname, start_frame, end_frame))

            for frame_name in sorted([int(frame_name) for frame_name in database[videoname]['frames'].keys()])[start_frame:end_frame]:
                if self.max_frames != 0 and num_video_frames >= self.max_frames:
                    break

                frame = database[videoname]['frames'][str(frame_name)]

                if "annos" not in frame:
                    continue

                self.frames[frame_index] = [videoname, "{0:05d}.jpg".format(frame['rgb_image_id'])]
                self.video_id_frame_id_to_frame_index[(videoname, "{0:05d}.jpg".format(frame['rgb_image_id']))] = frame_index

                image = self.processor(Image.open(os.path.join(THIS_DIR, "../data/rgb-images", videoname, "{0:05d}.jpg".format(frame['rgb_image_id']))))
                self.pixel_values[frame_index] = image['pixel_values'][0]
                self.pixel_mask[frame_index] = image['pixel_mask'][0]

                # Extract labels and box coordinate for each box in the frame.
                frame_labels = torch.zeros(size=(self.num_queries, NUM_CLASSES + 1), dtype=torch.float32)
                frame_boxes = torch.zeros(size=(self.num_queries, 4))
                for bounding_box_index, bounding_box in enumerate(frame['annos']):
                    frame_boxes[bounding_box_index] = torch.tensor(frame['annos'][bounding_box]['box'])
                    frame_labels[bounding_box_index, -1] = 1  # Set the last class to 1 to indicate that there is an object box here.
                    for label_type in LABEL_TYPES:
                        for label_id in frame['annos'][bounding_box][label_type + '_ids']:
                            if int(label_id) not in ORIGINAL_LABEL_MAPPING[label_type]:
                                continue
                            frame_labels[bounding_box_index][ORIGINAL_LABEL_MAPPING[label_type][int(label_id)][0]] = True

                self.labels[frame_index] = frame_labels
                self.boxes[frame_index] = frame_boxes

                frame_index += 1
                num_video_frames += 1

            logging.debug("Frames loaded: {0}".format(frame_index))
        logging.info("Total frames loaded for all videos: {0}".format(frame_index))

        self.pixel_values = torch.from_numpy(self.pixel_values)
        self.pixel_mask = torch.from_numpy(self.pixel_mask)

    def get_frame_and_video_names(self, frame_id):
        return self.frames[frame_id]

    def image_height(self):
        return int(IMAGE_HEIGHT * self.image_resize)

    def image_width(self):
        return int(IMAGE_WIDTH * self.image_resize)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frame_ids[index], self.pixel_values[index], self.pixel_mask[index], self.labels[index], self.boxes[index]
