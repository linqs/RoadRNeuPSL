import json
import logging
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.io

from torch.utils.data import Dataset

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

IMAGE_HEIGHT = 960
IMAGE_WIDTH = 1280
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

LABEL_TYPES = ['agent', 'action', 'loc']
NUM_CLASSES = 41

LABEL_MAPPING = {
    "agent": {
        0: [0, "Ped"],
        1: [1, "Car"],
        2: [2, "Cyc"],
        3: [3, "Mobike"],
        5: [4, "MedVeh"],
        6: [5, "LarVeh"],
        7: [6, "Bus"],
        8: [7, "EmVeh"],
        9: [8, "TL"],
        10: [9, "OthTL"]
    },
    "action": {
        0: [10, "Red"],
        1: [11, "Amber"],
        2: [12, "Green"],
        3: [13, "MovAway"],
        4: [14, "MovTow"],
        5: [15, "Mov"],
        7: [16, "Brake"],
        8: [17, "Stop"],
        9: [18, "IncatLft"],
        10: [19, "IncatRht"],
        11: [20, "HazLit"],
        12: [21, "TurLft"],
        13: [22, "TurRht"],
        16: [23, "Ovtak"],
        17: [24, "Wait2X"],
        18: [25, "XingFmLft"],
        19: [26, "XingFmRht"],
        20: [27, "Xing"],
        21: [28, "PushObj"]
    },
    "loc": {
        0: [29, "VehLane"],
        1: [30, "OutgoLane"],
        2: [31, "OutgoCycLane"],
        3: [32, "IncomLane"],
        4: [33, "IncomCycLane"],
        5: [34, "Pav"],
        6: [35, "LftPav"],
        7: [36, "RhtPav"],
        8: [37, "Jun"],
        9: [38, "xing"],
        10: [39, "BusStop"],
        11: [40, "parking"]
    }
}


class RoadRDataset(Dataset):
    def __init__(self,
                 videos,
                 data_path,
                 image_resize,
                 num_queries,
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

        self.frames = None
        self.frame_ids = None
        self.images = None
        self.labels = None
        self.boxes = None

        self.transforms = transforms.Compose([
            transforms.Resize(size=(int(IMAGE_HEIGHT * self.image_resize), int(IMAGE_WIDTH * self.image_resize)), antialias=True),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)])

        self.load_data()
        self.frame_ids_mapping = {frame_id: i for i, frame_id in enumerate(self.frame_ids.tolist())}

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

        # TODO(Charles): By pre allocating the max bounding boxes per frame, we are wasting a lot of memory.
        # Additionally, in evaluation we will have bounding boxes that are all zeros.
        self.frames = np.empty(shape=(num_frames, 2), dtype=object)  # (video_id, frame_id)
        self.frame_ids = torch.arange(num_frames, dtype=torch.int64)
        self.images = torch.empty(size=(num_frames, 3, int(IMAGE_HEIGHT * self.image_resize), int(IMAGE_WIDTH * self.image_resize)), dtype=torch.float32)
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

                self.images[frame_index] = self.transforms(
                    torchvision.io.read_image(
                        os.path.join(THIS_DIR, "../data/rgb-images", videoname, "{0:05d}.jpg".format(frame['rgb_image_id']))
                    ).type(torch.float32))
                self.frames[frame_index] = [videoname, "{0:05d}.jpg".format(frame['rgb_image_id'])]

                # Extract labels and box coordinate for each box in the frame.
                frame_labels = torch.zeros(size=(self.num_queries, NUM_CLASSES + 1), dtype=torch.float32)
                frame_boxes = torch.zeros(size=(self.num_queries, 4))
                for bounding_box_index, bounding_box in enumerate(frame['annos']):
                    frame_boxes[bounding_box_index] = torch.tensor(frame['annos'][bounding_box]['box'])
                    frame_labels[bounding_box_index, -1] = 1  # Set the last class to 1 to indicate that there is an object box here.
                    for label_type in LABEL_TYPES:
                        for label_id in frame['annos'][bounding_box][label_type + '_ids']:
                            if int(label_id) not in LABEL_MAPPING[label_type]:
                                continue
                            frame_labels[bounding_box_index][LABEL_MAPPING[label_type][int(label_id)][0]] = True

                self.labels[frame_index] = frame_labels
                self.boxes[frame_index] = frame_boxes

                frame_index += 1
                num_video_frames += 1

            logging.debug("Frames loaded: {0}".format(frame_index))
        logging.info("Total frames loaded for all videos: {0}".format(frame_index))

    def get_frame_and_video_names(self, frame_id):
        return self.frames[frame_id]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frame_ids[index], self.images[index], self.labels[index], self.boxes[index]
