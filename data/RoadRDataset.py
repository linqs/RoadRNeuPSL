import json
import logging
import numpy as np
import os
import torch
import torch.utils as torch_utils
import torchvision.transforms as transforms
import torchvision.io

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

IMAGE_HEIGHT = 960
IMAGE_WIDTH = 1280
IMAGE_RESIZE = 1.0
IMAGE_SIZE = (int(IMAGE_HEIGHT * IMAGE_RESIZE), int(IMAGE_WIDTH * IMAGE_RESIZE))
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

LABEL_TYPES = ['agent', 'action', 'loc']
LABEL_TYPE_OFFSETS = {
    'agent': 0,
    'action': 10,
    'loc': 29
}
NUM_CLASSES = 41
MAX_BOUNDING_BOXES_PER_FRAME = 25


class RoadRDataset(torch_utils.data.Dataset):
    def __init__(self, videos, data_path, max_frames=None):
        self.labeled_videos = videos
        self.data_path = data_path
        self.max_frames = max_frames

        self.frames = None
        self.images = None
        self.labels = None
        self.boxes = None

        self.transforms = transforms.Compose([
            transforms.Resize(size=(int(IMAGE_HEIGHT * IMAGE_RESIZE), int(IMAGE_WIDTH * IMAGE_RESIZE)), antialias=True),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)])

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

            logging.debug("Counting frames for video: {0}".format(videoname))

            for frame_name in database[videoname]['frames']:
                if self.max_frames is not None and num_frames >= self.max_frames:
                    break

                frame = database[videoname]['frames'][frame_name]

                if "annos" not in frame:
                    continue

                num_frames += 1
        logging.debug("Total frames counted in all videos: {0}".format(num_frames))

        self.frames = np.empty(shape=(num_frames, 2), dtype=object)  # (video_id, frame_id)
        self.images = torch.empty(size=(num_frames, 3, int(IMAGE_HEIGHT * IMAGE_RESIZE), int(IMAGE_WIDTH * IMAGE_RESIZE)), dtype=torch.float32)
        self.labels = torch.empty(size=(num_frames, MAX_BOUNDING_BOXES_PER_FRAME, NUM_CLASSES + 1), dtype=torch.bool)
        self.boxes = torch.empty(size=(num_frames, MAX_BOUNDING_BOXES_PER_FRAME, 4), dtype=torch.float32)

        logging.info("Loading frames for all videos.")
        frame_index = 0
        for videoname in sorted(database.keys()):
            if not videoname in self.labeled_videos:
                continue

            logging.debug("Loading frames for video: {0}".format(videoname))

            for frame_name in database[videoname]['frames']:
                if self.max_frames is not None and frame_index >= self.max_frames:
                    break

                frame = database[videoname]['frames'][frame_name]

                if "annos" not in frame:
                    continue

                self.images[frame_index] = self.transforms(torchvision.io.read_image(os.path.join(THIS_DIR, "../data/rgb-images", videoname, "{0:05d}.jpg".format(frame['rgb_image_id']))).type(torch.float32))
                self.frames[frame_index] = [videoname, str(frame['rgb_image_id'])]

                # Extract labels and box coordinate for each box in the frame.
                frame_labels = torch.zeros(size=(MAX_BOUNDING_BOXES_PER_FRAME, NUM_CLASSES + 1), dtype=torch.bool)
                frame_boxes = torch.zeros(size=(MAX_BOUNDING_BOXES_PER_FRAME, 4))
                for bounding_box_index, bounding_box in enumerate(frame['annos']):
                    frame_boxes[bounding_box_index] = torch.tensor(frame['annos'][bounding_box]['box'])
                    frame_labels[bounding_box_index, -1] = 1  # Set the last class to 1 to indicate that there is an object box here.
                    for label_type in LABEL_TYPES:
                        for label_id in frame_labels[bounding_box_index][frame['annos'][bounding_box][label_type + '_ids']]:
                            frame_labels[bounding_box_index][int(label_id) + LABEL_TYPE_OFFSETS[label_type]] = True

                self.labels[frame_index] = frame_labels
                self.boxes[frame_index] = frame_boxes

                frame_index += 1

            logging.debug("Frames loaded: {0}".format(frame_index))
        logging.info("Total frames loaded for all videos: {0}".format(frame_index))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index], self.images[index], self.labels[index], self.boxes[index]
