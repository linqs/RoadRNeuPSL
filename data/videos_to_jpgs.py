import argparse
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger

from utils import BASE_DATA_DIR
from utils import BASE_RGB_IMAGES_DIR

VIDEOS_DIR = os.path.join(BASE_DATA_DIR, "videos")


def video_to_jpgs(video_path):
    video_filename = os.path.basename(video_path)[:-4]
    os.makedirs(os.path.join(BASE_RGB_IMAGES_DIR, video_filename), exist_ok=True)

    if len(os.listdir(os.path.join(BASE_RGB_IMAGES_DIR, video_filename))) > 0:
        logging.info("Video {0} already processed: Skipping".format(video_filename))
        return

    command = 'ffmpeg  -i {} -q:v 1 {}/%05d.jpg'.format(video_path, os.path.join(BASE_RGB_IMAGES_DIR, video_filename))

    logging.info("Running command: {0}".format(command))
    os.system(command)


def main(arguments):
    logger.initLogging(arguments.log_level)

    for video_filename in os.listdir(VIDEOS_DIR):
        if not video_filename.endswith(".mp4"):
            continue

        logging.info("Processing video {0}".format(video_filename))
        video_to_jpgs(os.path.join(VIDEOS_DIR, video_filename))


def _load_args():
    parser = argparse.ArgumentParser(description="RoadR Dataset Processor")

    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default="INFO",
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    main(_load_args())
