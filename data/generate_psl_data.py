import argparse
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logger
import utils

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
SYMBOLIC_DATA_DIR = os.path.join(THIS_DIR, 'symbolic-data')

LOGGING_LEVEL = logging.INFO
CONFIG_FILENAME = 'config.json'

NUM_CLASSES = 45


def generate_experiment(experiment_dir, tube_size):
    utils.make_dir(experiment_dir)

    entity_targets = []
    for tube_index in range(tube_size):
        for class_index in range(NUM_CLASSES):
            entity_targets.append([tube_index, class_index])

    next_frame = []
    for tube_index in range(tube_size - 1):
        next_frame.append([tube_index, tube_index + 1])

    hard_co_occurrence = []
    raw_hard_co_occurrence = utils.load_csv_file(os.path.join(THIS_DIR, 'constraints', 'hard-co-occurrence.csv'))
    raw_hard_co_occurrence = [row[1:] for row in raw_hard_co_occurrence[1:]]
    for tube_index in range(tube_size):
        for index_i in range(len(raw_hard_co_occurrence)):
            for index_j in range(len(raw_hard_co_occurrence[index_i])):
                if index_i == index_j:
                    continue
                hard_co_occurrence.append([tube_index, index_i, index_j, int(raw_hard_co_occurrence[index_i][index_j])])

    utils.write_psl_file(os.path.join(experiment_dir, 'entity_targets.psl'), entity_targets)
    utils.write_psl_file(os.path.join(experiment_dir, 'next_frame.psl'), next_frame)
    utils.write_psl_file(os.path.join(experiment_dir, 'hard_co_occurrence.psl'), hard_co_occurrence)


def _load_args():
    parser = argparse.ArgumentParser(description='Generate Road-R PSL data.')

    parser.add_argument('--tube-size', dest='tubeSize',
                        action='store', type=int, default=4,
                        help='The size of the tube used to generate the '
                             'symbolic data.')

    arguments = parser.parse_args()

    return arguments


def main(arguments):
    logger.initLogging(LOGGING_LEVEL)
    logging.info("Beginning Road-R PSL data generation.")
    logging.debug("Arguments: %s" % (arguments,))

    logging.info("Generating PSL data with tube size: %d" % (arguments.tubeSize,))
    experiment_dir = os.path.join(SYMBOLIC_DATA_DIR, "experiment::tube-size-" + str(arguments.tubeSize))
    generate_experiment(experiment_dir, arguments.tubeSize)


if __name__ == '__main__':
    main(_load_args())
