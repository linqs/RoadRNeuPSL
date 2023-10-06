import argparse
import logging
import torch

import logger

STANDARD_EXPERIMENT_OPTIONS = {
    "runtime.log.level": "TRACE",
    "runtime.db.intids": "true",
    "runtime.learn": "true",
    "runtime.learn.method": "Energy",
    "runtime.inference.deep.batching": "true",
    "runtime.inference.method": "GurobiInference",
    "weightlearning.inference": "GurobiInference",
    "inference.normalize": "false",
    "gradientdescent.scalestepsize": "false",
    "gradientdescent.trainingcomputeperiod": "5",
    "gradientdescent.stopcomputeperiod": "5",
    "gradientdescent.numsteps": "50",
    "gradientdescent.runfulliterations": "true",
    "gradientdescent.batchgenerator": "NeuralBatchGenerator"
}

def run_neupsl(arguments):


def main(arguments):
    logger.initLogging(arguments.log_level)

    logging.info("Beginning pre-training.")
    logging.info("Beginning pre-training.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    run_neupsl(arguments)


def _load_args():
    parser = argparse.ArgumentParser(description="RoadR Task 1 Pre-Training Network")

    parser.add_argument("--task", dest="task", type=str, choices=["task1", "task2"])
    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default="INFO",
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--image-resize", dest="image_resize",
                        action="store", type=float, default=1.0,
                        help="Resize factor for all images.")
    parser.add_argument("--max-frames", dest="max_frames",
                        action="store", type=int, default=0,
                        help="Maximum number of frames to use from each videos. Default is 0, which uses all frames.")
    parser.add_argument("--hyperparameter-search", dest="hyperparameter_search",
                        action="store", type=bool, default=False,
                        help="Run hyperparameter search.")
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint",
                        action="store", type=bool, default=False,
                        help="Resume training from the most recent checkpoint.")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
