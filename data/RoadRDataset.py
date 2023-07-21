import json
import numpy as np
import os
import torch
import torch.utils as torch_utils

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

INPUT_TYPE = 'rgb'
LABEL_TYPES = ['agent', 'action', 'loc']
LABEL_TYPE_OFFSETS = {
    'agent': 0,
    'action': 10,
    'loc': 29
}
NUM_CLASSES = 41
MAX_BOUNDING_BOXES_PER_FRAME = 25


class RoadRDataset(torch_utils.data.Dataset):
    def __init__(self, videos, data_path):
        self.labeled_videos = videos
        self.data_path = data_path
        self.frames = None
        self.labels = None
        self.boxes = None

        self.load_data()

    def load_data(self):
        with open(self.data_path, 'r') as data_file:
            json_data = json.load(data_file)

        database = json_data['db']

        # Count the number of frames to preallocate memory.
        num_frames = 0
        for videoname in sorted(database.keys()):
            if not videoname in self.labeled_videos:
                continue

            for frame_name in database[videoname]['frames']:
                frame = database[videoname]['frames'][frame_name]

                # 127 frames in the train validation set not annotated.
                if "annos" not in frame:
                    continue

                num_frames += 1

        self.frames = np.empty(shape=(num_frames, 2), dtype=object)  # (video_id, frame_id)
        self.labels = np.empty(shape=(num_frames, MAX_BOUNDING_BOXES_PER_FRAME, NUM_CLASSES), dtype=np.int8)
        self.boxes = np.empty(shape=(num_frames, MAX_BOUNDING_BOXES_PER_FRAME, 4), dtype=np.float32)

        frame_index = 0
        for videoname in sorted(database.keys()):
            if not videoname in self.labeled_videos:
                continue

            for frame_name in database[videoname]['frames']:
                frame = database[videoname]['frames'][frame_name]

                # 127 frames in the train validation set not annotated.
                if "annos" not in frame:
                    continue

                self.frames[frame_index] = [videoname, str(frame['rgb_image_id'])]

                # Extract labels and box coordinate for each box in the frame.
                frame_labels = np.zeros(shape=(MAX_BOUNDING_BOXES_PER_FRAME, NUM_CLASSES))
                frame_boxes = np.zeros(shape=(MAX_BOUNDING_BOXES_PER_FRAME, 4))
                for bounding_box_index, bounding_box in enumerate(frame['annos']):
                    frame_boxes[bounding_box_index] = frame['annos'][bounding_box]['box']
                    for label_type in LABEL_TYPES:
                        for label_id in frame_labels[bounding_box_index][frame['annos'][bounding_box][label_type + '_ids']]:
                            frame_labels[bounding_box_index][int(label_id) + LABEL_TYPE_OFFSETS[label_type]] = 1

                self.labels[frame_index] = frame_labels
                self.boxes[frame_index] = frame_boxes

                frame_index += 1

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index], self.labels[index], self.boxes[index]
