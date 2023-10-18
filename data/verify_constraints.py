import csv
import json
import logging
import os
import sys

import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger

from utils import ORIGINAL_LABEL_MAPPING as LABEL_MAPPING

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_FILE_NAME = "road_trainval_v1.0.json"
OUT_FILE_NAME = "verify_constraints_results.csv"

ALL_VIDEOS = True
LABELED_VIDEOS = ["2014-07-14-14-49-50_stereo_centre_01",
                  "2015-02-03-19-43-11_stereo_centre_04",
                  "2015-02-24-12-32-19_stereo_centre_04"]

LABEL_TYPES = ["agent", "action", "loc"]
NUM_CLASSES = 41


def load_labels(data_path):
    with open(data_path, 'r') as data_file:
        json_data = json.load(data_file)

    database = json_data['db']

    labels = []
    index_mapping = {}
    for videoname in sorted(database.keys()):
        if not ALL_VIDEOS:
            if not videoname in LABELED_VIDEOS:
                continue

        for frame_name in database[videoname]['frames']:
            frame = database[videoname]['frames'][frame_name]

            if "annos" not in frame:
                continue

            bb_index = 0
            for bounding_box in frame['annos']:
                labels.append([0] * NUM_CLASSES)
                index_mapping[len(index_mapping)] = (videoname, frame_name, bb_index)
                for label_type in LABEL_TYPES:
                    for label_id in frame['annos'][bounding_box][label_type + '_ids']:
                        if label_id not in LABEL_MAPPING[label_type]:
                            continue
                        labels[-1][LABEL_MAPPING[label_type][int(label_id)][0]] = 1
                bb_index += 1
    return labels, index_mapping


def main():
    logger.initLogging("INFO")

    labels, index_mapping = load_labels(os.path.join(THIS_DIR, DATA_FILE_NAME))

    label_count = numpy.zeros((NUM_CLASSES, NUM_CLASSES))
    non_traffic_light_contains_action = 0
    non_traffic_light = 0
    image_index = 0

    for label in labels:
        if sum(label[:8]) > 0:
            if sum(label[10:29]) >= 1.0:
                non_traffic_light_contains_action += 1
            else:
                logging.info("Non-traffic light: %s", index_mapping[image_index])
                logging.info(label)
            non_traffic_light += 1
        for index in range(len(label)):
            if label[index] == 1:
                label_count[index] += numpy.array(label)
        image_index += 1

    logging.info("Non-traffic light contains action: %d", non_traffic_light_contains_action)
    logging.info("Non-traffic light: %d", non_traffic_light)

    with open(OUT_FILE_NAME, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(label_count.tolist())

    below_lower_threshold = []
    above_upper_threshold = []
    lower_threshold = 0.01
    upper_threshold = 0.50
    for index in range(len(label_count)):
        below_lower_label = [0] * len(label_count)
        above_upper_label = [0] * len(label_count)
        for index2 in range(index, len(label_count[index])):
            if index == index2 or label_count[index][index2] == 0:
                continue
            if label_count[index][index2] / label_count[index][index] < lower_threshold:
                below_lower_label[index2] = 1
            if label_count[index][index2] / label_count[index][index] > upper_threshold:
                above_upper_label[index2] = 1
        below_lower_threshold.append(below_lower_label)
        above_upper_threshold.append(above_upper_label)
    with open("below_lower_threshold.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(below_lower_threshold)
    with open("above_upper_threshold.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(above_upper_threshold)



if __name__ == '__main__':
    main()