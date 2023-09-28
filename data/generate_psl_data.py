import argparse
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger
import utils

from utils import NUM_CLASSES
from utils import NUM_NEUPSL_QUERIES

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
PSL_DATA_DIR = os.path.join(THIS_DIR, "psl-data")

LOGGING_LEVEL = logging.INFO
CONFIG_FILENAME = "config.json"

AGENT_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
BOUNDING_BOX_CLASSES = [41, 42, 43, 44, 45]


def generate_experiment(experiment_dir, tube_size):
    utils.make_dir(experiment_dir)

    entity_data_map = []
    for tube_index in range(tube_size):
        for bounding_box_index in range(NUM_NEUPSL_QUERIES):
            entity_data_map.append([tube_index, bounding_box_index])

    agent_classes = []
    for agent_class in AGENT_CLASSES:
        agent_classes.append([agent_class])

    bounding_box_classes = []
    for bounding_box_class in BOUNDING_BOX_CLASSES:
        bounding_box_classes.append([bounding_box_class])

    entity_targets = []
    for tube_index in range(tube_size):
        for bounding_box_index in range(NUM_NEUPSL_QUERIES):
            for class_index in range(NUM_CLASSES + len(BOUNDING_BOX_CLASSES)):
                entity_targets.append([tube_index, bounding_box_index, class_index])

    hard_co_occurrence = []
    raw_hard_co_occurrence = utils.load_csv_file(os.path.join(THIS_DIR, "constraints", "hard-co-occurrence.csv"))
    raw_hard_co_occurrence = [row[1:] for row in raw_hard_co_occurrence[1:]]
    for index_i in range(len(raw_hard_co_occurrence)):
        for index_j in range(len(raw_hard_co_occurrence)):
            hard_co_occurrence.append([index_i, index_j, int(raw_hard_co_occurrence[index_i][index_j])])

    utils.write_psl_file(os.path.join(experiment_dir, "entity-data-map.txt"), entity_data_map)
    utils.write_psl_file(os.path.join(experiment_dir, "classes-agent.txt"), agent_classes)
    utils.write_psl_file(os.path.join(experiment_dir, "classes-bounding-box.txt"), bounding_box_classes)
    utils.write_psl_file(os.path.join(experiment_dir, "entity-targets.txt"), entity_targets)
    utils.write_psl_file(os.path.join(experiment_dir, "hard-co-occurrence.txt"), hard_co_occurrence)


def _load_args():
    parser = argparse.ArgumentParser(description="Generate Road-R PSL data.")

    parser.add_argument("--tube-size", dest="tubeSize",
                        action="store", type=int, default=32,
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
