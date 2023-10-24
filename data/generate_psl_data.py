import argparse
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger
import utils

from utils import load_constraint_file
from utils import HARD_CONSTRAINTS_PATH
from utils import SOFT_CONSTRAINTS_PATH
from utils import UNCOMMON_CONSTRAINTS_PATH
from utils import NUM_CLASSES
from utils import NUM_NEUPSL_QUERIES

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
PSL_DATA_DIR = os.path.join(THIS_DIR, "psl-data")

LOGGING_LEVEL = logging.INFO
CONFIG_FILENAME = "config.json"

AGENT_CLASSES = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
ACTION_CLASSES = [[10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28]]
BOX_CONFIDENCE_CLASS = [[41]]
BOUNDING_BOX_CLASSES = [[42], [43], [44], [45]]
LOCATION_CLASSES = [[29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40]]


def generate_experiment(experiment_dir, tube_size):
    os.makedirs(experiment_dir, exist_ok=True)

    entity_data_map = []
    for tube_index in range(tube_size):
        for bounding_box_index in range(NUM_NEUPSL_QUERIES):
            entity_data_map.append([tube_index, bounding_box_index])

    entity_targets = []
    linked_frames = []
    same_box_targets = []
    same_corner_targets = []
    for tube_index in range(tube_size):
        if tube_index < tube_size - 1:
            linked_frames.append([tube_index, tube_index + 1])
            linked_frames.append([tube_index + 1, tube_index])

        for bounding_box_index_i in range(NUM_NEUPSL_QUERIES):
            for class_index in range(NUM_CLASSES + len(BOX_CONFIDENCE_CLASS) + len(BOUNDING_BOX_CLASSES)):
                entity_targets.append([tube_index, bounding_box_index_i, class_index])

            if tube_index < tube_size - 1:
                for bounding_box_index_j in range(NUM_NEUPSL_QUERIES):
                    same_box_targets.append([tube_index, tube_index + 1, bounding_box_index_i, bounding_box_index_j])
                    same_box_targets.append([tube_index + 1, tube_index, bounding_box_index_i, bounding_box_index_j])
                    for corner_index in range(4):
                        same_corner_targets.append([tube_index, tube_index + 1, bounding_box_index_i, bounding_box_index_j, corner_index])
                        same_corner_targets.append([tube_index + 1, tube_index, bounding_box_index_i, bounding_box_index_j, corner_index])


    hard_co_occurrence = load_constraint_file(HARD_CONSTRAINTS_PATH)
    soft_co_occurrence = load_constraint_file(SOFT_CONSTRAINTS_PATH)
    uncommon_co_occurrence = load_constraint_file(UNCOMMON_CONSTRAINTS_PATH)

    utils.write_psl_file(os.path.join(experiment_dir, "entity-data-map.txt"), entity_data_map)
    utils.write_psl_file(os.path.join(experiment_dir, "classes-agent.txt"), AGENT_CLASSES)
    utils.write_psl_file(os.path.join(experiment_dir, "classes-action.txt"), ACTION_CLASSES)
    utils.write_psl_file(os.path.join(experiment_dir, "classes-bounding-box.txt"), BOUNDING_BOX_CLASSES)
    utils.write_psl_file(os.path.join(experiment_dir, "classes-box-confidence.txt"), BOX_CONFIDENCE_CLASS)
    utils.write_psl_file(os.path.join(experiment_dir, "classes-location.txt"), LOCATION_CLASSES)
    utils.write_psl_file(os.path.join(experiment_dir, "entity-targets.txt"), entity_targets)
    utils.write_psl_file(os.path.join(experiment_dir, "linked-frame.txt"), linked_frames)
    utils.write_psl_file(os.path.join(experiment_dir, "same-box-targets.txt"), same_box_targets)
    utils.write_psl_file(os.path.join(experiment_dir, "same-corner-targets.txt"), same_corner_targets)
    utils.write_psl_file(os.path.join(experiment_dir, "hard-co-occurrence.txt"), hard_co_occurrence)
    utils.write_psl_file(os.path.join(experiment_dir, "soft-co-occurrence.txt"), soft_co_occurrence)
    utils.write_psl_file(os.path.join(experiment_dir, "uncommon-co-occurrence.txt"), uncommon_co_occurrence)


def _load_args():
    parser = argparse.ArgumentParser(description="Generate Road-R PSL data.")

    parser.add_argument("--tube-size", dest="tubeSize",
                        action="store", type=int, default=4,
                        help="The size of the tube used to generate the "
                             "symbolic data.")

    arguments = parser.parse_args()

    return arguments


def main(arguments):
    logger.initLogging(LOGGING_LEVEL)
    logging.info("Beginning Road-R PSL data generation.")
    logging.debug("Arguments: %s" % (arguments,))

    logging.info("Generating PSL data with tube size: %d" % (arguments.tubeSize,))
    experiment_dir = os.path.join(PSL_DATA_DIR, "experiment::tube-size-" + str(arguments.tubeSize))
    generate_experiment(experiment_dir, arguments.tubeSize)


if __name__ == "__main__":
    main(_load_args())
