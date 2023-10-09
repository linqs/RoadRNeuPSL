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
from utils import NEURAL_TRAINED_MODEL_DIR
from utils import NEUPSL_TRAINED_MODEL_DIR
from utils import NEUPSL_VALID_INFERENCE_DIR
from utils import NEUPSL_TEST_INFERENCE_DIR

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
    base_out_dir = os.path.join(BASE_RESULTS_DIR, arguments.task)
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file.
    dataset_json_path = os.path.join(PSL_MODELS_DIR, "roadr.json")

    psl_json = None
    with open(dataset_json_path, "r") as file:
        psl_json = json.load(file)
    original_options = psl_json["options"]

    # Update the options.
    psl_json.update({"options": {**original_options,
                                 **STANDARD_EXPERIMENT_OPTIONS}})

    set_log_options(psl_json, arguments)
    set_neural_predicate_options(psl_json, arguments)

    learning_out_dir = os.path.join(base_out_dir, "neupsl_learning")
    os.makedirs(learning_out_dir, exist_ok=True)

    if not arguments.no_learning:
        # Run NeuPSL training.
        set_runtime_task(psl_json, "true", "false")

        write_neupsl_json(psl_json)

        print("Running NeuPSL learning for: {}.".format(base_out_dir))
        exit_code = os.system("cd {} && ./run.sh > {}/learning_out.txt 2> {}/learning_out.err".format(BASE_CLI_DIR, learning_out_dir, learning_out_dir))

        if exit_code != 0:
            print("Experiment failed: {}.".format(base_out_dir))
            exit()

    inference_out_dir = None
    if arguments.test_inference and arguments.use_neural_trained_model:
        inference_out_dir = os.path.join(base_out_dir, NEUPSL_TEST_INFERENCE_DIR, NEURAL_TRAINED_MODEL_DIR)
    elif arguments.test_inference and (not arguments.use_neural_trained_model):
        inference_out_dir = os.path.join(base_out_dir, NEUPSL_TEST_INFERENCE_DIR, NEUPSL_TRAINED_MODEL_DIR)
    elif (not arguments.test_inference) and arguments.use_neural_trained_model:
        inference_out_dir = os.path.join(base_out_dir, NEUPSL_VALID_INFERENCE_DIR, NEURAL_TRAINED_MODEL_DIR)
    elif (not arguments.test_inference) and (not arguments.use_neural_trained_model):
        inference_out_dir = os.path.join(base_out_dir, NEUPSL_VALID_INFERENCE_DIR, NEUPSL_TRAINED_MODEL_DIR)

    os.makedirs(inference_out_dir, exist_ok=True)

    if not arguments.no_inference:
        # Run NeuPSL inference.
        set_runtime_task(psl_json, "false", "true")
        set_inference_split(psl_json, arguments)

        if arguments.task == "task2":
            psl_json["predicates"]["Label/3"]["options"]["Integer"] = "true"

        write_neupsl_json(psl_json)

        print("Running NeuPSL inference for: {}.".format(inference_out_dir))
        exit_code = os.system("cd {} && ./run.sh > {}/inference_out.txt 2> {}/inference_out.err".format(BASE_CLI_DIR, inference_out_dir, inference_out_dir))

        if exit_code != 0:
            print("Experiment failed: {}.".format(inference_out_dir))
            exit()

        # Save the output and json file.
        os.system("cp {} {}".format(os.path.join(BASE_CLI_DIR, "roadr.json"), inference_out_dir))
        os.system("cp -r {} {}".format(os.path.join(BASE_CLI_DIR, "inferred-predicates"), inference_out_dir))


def set_inference_split(psl_json, arguments):
    psl_json["predicates"]["Neural/3"]["options"]["inference_split"] = "TEST" if arguments.test_inference else "VALID"


def set_log_options(psl_json, arguments):
    psl_json["options"]["runtime.log.level"] = arguments.psl_log_level
    if arguments.gurobi_log:
        psl_json["options"]["gurobi.logtoconsole"] = "true"


def set_neural_predicate_options(psl_json, arguments):
    psl_json["predicates"]["Neural/3"]["options"]["task-name"] = arguments.task
    psl_json["predicates"]["Neural/3"]["options"]["image-resize"] = arguments.image_resize
    psl_json["predicates"]["Neural/3"]["options"]["max-frames"] = arguments.max_frames
    psl_json["predicates"]["Neural/3"]["options"]["evaluation-dir-name"] = arguments.evaluation_dir_name
    psl_json["predicates"]["Neural/3"]["options"]["use-neural-trained-model"] = arguments.use_neural_trained_model


def set_runtime_task(psl_json, learning, inference):
    psl_json["options"]["runtime.learn"] = learning
    psl_json["options"]["runtime.inference"] = inference


def write_neupsl_json(psl_json):
    # Write the options the json file.
    with open(os.path.join(BASE_CLI_DIR, "roadr.json"), "w") as file:
        json.dump(psl_json, file, indent=4)


def main(arguments):
    logger.initLogging(arguments.log_level)

    logging.info("NeuPSL.")
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
    parser.add_argument("--evaluation-dir-name", dest="evaluation_dir_name",
                        action="store", type=str, default="evaluation",
                        help="Name of the evaluation directory.")
    parser.add_argument("--gurobi-log", dest="gurobi_log",
                        action="store_true", default=False,
                        help="Turn on Gurobi logging.")
    parser.add_argument("--image-resize", dest="image_resize",
                        action="store", type=float, default=1.0,
                        help="Resize factor for all images.")
    parser.add_argument("--max-frames", dest="max_frames",
                        action="store", type=int, default=0,
                        help="Maximum number of frames to use from each videos. Default is 0, which uses all frames.")
    parser.add_argument("--no-learning", dest="no_learning",
                        action="store_true", default=False,
                        help="Turn off learning step.")
    parser.add_argument("--use-neural-trained-model", dest="use_neural_trained_model",
                        action="store_true", default=False,
                        help="Use the neural trained model.")
    parser.add_argument("--no-inference", dest="no_inference",
                        action="store_true", default=False,
                        help="Turn off inference step.")
    parser.add_argument("--test-inference", dest="test_inference",
                        action="store_true", default=False,
                        help="Run inference on the test set.")

    arguments = parser.parse_args()

    print("Arguments: %s" % (arguments,))

    return arguments


if __name__ == "__main__":
    main(_load_args())
