import csv
import json
import os

import numpy

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_FILE_NAME = 'road_trainval_v1.0.json'
LABEL_TYPES = ['agent', 'action', 'loc']
LABEL_TYPE_OFFSETS = {
    'agent': 0,
    'action': 10,
    'loc': 29
}
NUM_CLASSES = 41


def load_labels(data_path):
    with open(data_path, 'r') as data_file:
        json_data = json.load(data_file)

    database = json_data['db']

    labels = []
    for videoname in sorted(database.keys()):
        for frame_name in database[videoname]['frames']:
            frame = database[videoname]['frames'][frame_name]

            if "annos" not in frame:
                continue

            for bounding_box in frame['annos']:
                labels.append([0] * NUM_CLASSES)
                for label_type in LABEL_TYPES:
                    for label_id in frame['annos'][bounding_box][label_type + '_ids']:
                        labels[-1][int(label_id) + LABEL_TYPE_OFFSETS[label_type]] = 1
    return labels


def main():
    labels = load_labels(os.path.join(THIS_DIR, DATA_FILE_NAME))

    label_count = numpy.zeros((NUM_CLASSES, NUM_CLASSES))

    for label in labels:
        for index in range(len(label)):
            if label[index] == 1:
                label_count[index] += numpy.array(label)

    with open("verify-constraints-results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(label_count.tolist())


if __name__ == '__main__':
    main()
