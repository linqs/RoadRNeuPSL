import argparse
import json
import logging
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger
from utils import BASE_CLI_DIR
from utils import BASE_RESULTS_DIR
from utils import PSL_MODELS_DIR
from utils import NEUPSL_TRAINED_MODEL_DIR

STANDARD_EXPERIMENT_OPTIONS = {
    "runtime.log.level": "INFO",
    "runtime.db.intids": "true",
    "runtime.learn": "true",
    "runtime.learn.method": "Energy",
    "runtime.inference.deep.batching": "true",
    "runtime.inference.method": "GurobiInference",
    "weightlearning.inference": "GurobiInference",
    "gurobi.worklimit": 60,
    "inference.normalize": "false",
    "gradientdescent.scalestepsize": "false",
    "gradientdescent.stepsize": "1.0e-14",
    "gradientdescent.trainingcomputeperiod": "5",
    "gradientdescent.stopcomputeperiod": "5",
    "gradientdescent.numsteps": "50",
    "gradientdescent.runfulliterations": "true",
    "gradientdescent.batchgenerator": "NeuralBatchGenerator"
}


def run_neupsl(arguments):
    out_dir = os.path.join(BASE_RESULTS_DIR, arguments.task, NEUPSL_TRAINED_MODEL_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # Load the json file.
    dataset_json_path = os.path.join(PSL_MODELS_DIR, "roadr.json")

    psl_json = None
    with open(dataset_json_path, "r") as file:
        psl_json = json.load(file)
    original_options = psl_json["options"]

    # Update the options.
    psl_json.update({"options": {**original_options,
                                     **STANDARD_EXPERIMENT_OPTIONS}})
    psl_json["options"]["runtime.log.level"] = arguments.psl_log_level

    # Set the Neural predicate options.
    psl_json["predicates"]["Neural/3"]["options"]["task-name"] = arguments.task
    psl_json["predicates"]["Neural/3"]["options"]["image-resize"] = arguments.image_resize
    psl_json["predicates"]["Neural/3"]["options"]["max-frames"] = arguments.max_frames

    # Run NeuPSL training.
    psl_json["options"]["runtime.learn"] = "true"
    psl_json["options"]["runtime.inference"] = "false"

    write_neupsl_json(psl_json)

    print("Running NeuPSL training for: {}.".format(out_dir))
    exit_code = os.system("cd {} && ./run.sh > {}/learning_out.txt 2> {}/learning_out.err".format(BASE_CLI_DIR, out_dir, out_dir))

    if exit_code != 0:
        print("Experiment failed: {}.".format(out_dir))
        exit()

    # Run NeuPSL inference.
    psl_json["options"]["runtime.learn"] = "false"
    psl_json["options"]["runtime.inference"] = "true"

    if arguments.task == "task2":
        psl_json["predicates"]["Label/3"]["options"]["integer"] = "true"

    write_neupsl_json(psl_json)

    print("Running NeuPSL inference for: {}.".format(out_dir))
    exit_code = os.system("cd {} && ./run.sh > {}/inference_out.txt 2> {}/inference_out.err".format(BASE_CLI_DIR, out_dir, out_dir))

    if exit_code != 0:
        print("Experiment failed: {}.".format(out_dir))
        exit()

    # Save the output and json file.
    os.system("cp {} {}".format(os.path.join(BASE_CLI_DIR, "roadr.json"), out_dir))
    os.system("cp -r {} {}".format(os.path.join(BASE_CLI_DIR, "inferred-predicates"), out_dir))


def write_neupsl_json(psl_json):
    # Write the options the json file.
    with open(os.path.join(BASE_CLI_DIR, "roadr.json"), "w") as file:
        json.dump(psl_json, file, indent=4)
def main(arguments):
    logger.initLogging(arguments.log_level)

    logging.info("Beginning NeuPSL training.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    run_neupsl(arguments)


def _load_args():
    parser = argparse.ArgumentParser(description="RoadR Task NeuPSL Training Network")

    parser.add_argument("--task", dest="task",
                        action="store", type=str, choices=["task1", "task2"])
    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default="INFO",
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--psl-log-level", dest="psl_log_level",
                        action="store", type=str, default="INFO",
                        help="PSL logging level.", choices=["DEBUG", "INFO", "TRACE"])
    parser.add_argument("--image-resize", dest="image_resize",
                        action="store", type=float, default=1.0,
                        help="Resize factor for all images.")
    parser.add_argument("--max-frames", dest="max_frames",
                        action="store", type=int, default=0,
                        help="Maximum number of frames to use from each videos. Default is 0, which uses all frames.")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
