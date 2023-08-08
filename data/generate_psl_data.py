import argparse
import logging
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logger
import utils

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
SYMBOLIC_DATA_DIR = os.path.join(THIS_DIR, 'symbolic-data')

LOGGING_LEVEL = logging.DEBUG
CONFIG_FILENAME = 'config.json'

NUM_CLASSES = 45


def generate_split(split_dir, tube_size):
    if os.path.exists(os.path.join(split_dir, CONFIG_FILENAME)):
        logging.info(
            "Split directory %s already exists, skipping." % (split_dir,))
        return

    logging.info(
        "Split directory does not exist, generating: %s." % (split_dir,))
    utils.make_dir(split_dir)

    # Generate the symbolic data.
    entity_targets = []
    for tube_index in range(tube_size):
        for class_index in range(NUM_CLASSES):
            entity_targets.append([tube_index, class_index])

    next_frame = []
    for tube_index in range(tube_size - 1):
        next_frame.append([tube_index, tube_index + 1])

    utils.write_psl_file(os.path.join(split_dir, 'entity_targets.json'), entity_targets)
    utils.write_psl_file(os.path.join(split_dir, 'next_frame.json'), next_frame)


def _load_args():
    parser = argparse.ArgumentParser(description='Generate custom Road-R PSL '
                                                 'data.')

    parser.add_argument('--tube-size', dest='tubeSize',
                        action='store', type=int, default=4,
                        help='The size of the tube used to generate the '
                             'symbolic data.')

    parser.add_argument('--seed', dest='seed',
                        action='store', type=int, default=None,
                        help='Random seed.')

    parser.add_argument('--splits', dest='splits',
                        action='store', type=int, default=1,
                        help='The number of splits to generate.')

    arguments = parser.parse_args()

    if arguments.splits < 1:
        print("Number of splits must be >= 1, got: %d." % (arguments.splits,),
              file=sys.stderr)
        sys.exit(2)

    return arguments


def main(arguments):
    logger.initLogging(LOGGING_LEVEL)
    logging.info("Beginning Road-R PSL data generation.")
    logging.debug("Arguments: %s" % (arguments,))

    seed = arguments.seed
    if seed is None:
        seed = random.randrange(2 ** 32 - 1)

    logging.info("Random seed: %d." % (seed,))
    utils.seed_everything(seed)

    for split in range(arguments.splits):
        logging.info("Generating split %d." % (split,))

        # Generate the symbolic data.
        split_dir = os.path.join(SYMBOLIC_DATA_DIR,
                                 "experiment::tube-size-" + str(
                                     arguments.tubeSize), str(split).zfill(2))
        generate_split(split_dir, arguments.tubeSize)


if __name__ == '__main__':
    main(_load_args())
