import csv
import json
import os

import numpy


THIS_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_FILE_NAME = "road_trainval_v1.0.json"
OUT_FILE_NAME = "verify_constraints_results.csv"
LABEL_TYPES = ["agent", "action", "loc"]
NUM_CLASSES = 41

ALL_VIDEOS = True
LABELED_VIDEOS = ["2014-07-14-14-49-50_stereo_centre_01",
                  "2015-02-03-19-43-11_stereo_centre_04",
                  "2015-02-24-12-32-19_stereo_centre_04"]

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


def load_labels(data_path):
    with open(data_path, 'r') as data_file:
        json_data = json.load(data_file)

    database = json_data['db']

    labels = []
    for videoname in sorted(database.keys()):
        if not ALL_VIDEOS:
            if not videoname in LABELED_VIDEOS:
                continue

        for frame_name in database[videoname]['frames']:
            frame = database[videoname]['frames'][frame_name]

            if "annos" not in frame:
                continue

            for bounding_box in frame['annos']:
                labels.append([0] * NUM_CLASSES)
                for label_type in LABEL_TYPES:
                    for label_id in frame['annos'][bounding_box][label_type + '_ids']:
                        if label_id not in LABEL_MAPPING[label_type]:
                            continue
                        labels[-1][LABEL_MAPPING[label_type][int(label_id)][0]] = 1
    return labels


def main():
    labels = load_labels(os.path.join(THIS_DIR, DATA_FILE_NAME))

    label_count = numpy.zeros((NUM_CLASSES, NUM_CLASSES))

    for label in labels:
        for index in range(len(label)):
            if label[index] == 1:
                label_count[index] += numpy.array(label)

    with open(OUT_FILE_NAME, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(label_count.tolist())


if __name__ == '__main__':
    main()
