import argparse
import logging
import os
import sys

import torch

from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logger

from data.roadr_dataset import RoadRDataset
from models.trainer import Trainer
from utils import get_torch_device
from utils import load_csv_file
from utils import seed_everything
from utils import write_json_file
from utils import BASE_RESULTS_DIR
from utils import NEURAL_TRAINED_MODEL_DIR
from utils import NEURAL_TRAINED_MODEL_FILENAME
from utils import NEURAL_TRAINING_SUMMARY_FILENAME
from utils import NUM_CLASSES
from utils import SEED
from utils import TRAIN_VALIDATION_DATA_PATH
from utils import VIDEO_PARTITIONS


DEFAULT_PARAMETERS = {
    "learning-rate": 1.0e-5,
    "weight-decay": 1.0e-5,
    "batch-size": 2,
    "step-size": 500,
    "gamma": 1.0,
    "pretrained": "facebook/detr-resnet-101",
    "revision": "no_timm"
}


def build_model():
    if DEFAULT_PARAMETERS["revision"] == "no_timm":
        return DetrForObjectDetection.from_pretrained(DEFAULT_PARAMETERS["pretrained"],
                                                      num_labels=NUM_CLASSES,
                                                      revision=DEFAULT_PARAMETERS["revision"],
                                                      ignore_mismatched_sizes=True).to(get_torch_device())

    return DetrForObjectDetection.from_pretrained(DEFAULT_PARAMETERS["pretrained"],
                                                  num_labels=NUM_CLASSES,
                                                  ignore_mismatched_sizes=True).to(get_torch_device())


def run_setting(arguments, train_dataset, validation_dataset, parameters, parameters_string):
    if os.path.isfile(os.path.join(BASE_RESULTS_DIR, arguments.task, parameters_string, NEURAL_TRAINING_SUMMARY_FILENAME)) and not arguments.resume_from_checkpoint:
        logging.info("Skipping training for %s, already exists." % (parameters_string,))
        return float(load_csv_file(os.path.join(BASE_RESULTS_DIR, arguments.task, parameters_string, NEURAL_TRAINING_SUMMARY_FILENAME))[1][0])

    train_dataloader = DataLoader(train_dataset, batch_size=parameters["batch-size"], shuffle=True,
                                  num_workers=int(os.cpu_count()) - 2, prefetch_factor=4, persistent_workers=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=parameters["batch-size"], shuffle=True,
                                       num_workers=int(os.cpu_count()) - 2, prefetch_factor=4, persistent_workers=True)

    model = build_model()

    if arguments.resume_from_checkpoint:
        logging.info("Loading model from checkpoint: %s" % (os.path.join(BASE_RESULTS_DIR, arguments.task, parameters_string, NEURAL_TRAINED_MODEL_FILENAME),))
        model.load_state_dict(torch.load(os.path.join(BASE_RESULTS_DIR, arguments.task, parameters_string, NEURAL_TRAINED_MODEL_FILENAME), map_location=get_torch_device()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning-rate"], weight_decay=parameters["weight-decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameters["step-size"], gamma=parameters["gamma"])

    trainer = Trainer(model, optimizer, lr_scheduler, get_torch_device(), os.path.join(BASE_RESULTS_DIR, arguments.task, parameters_string))
    trainer.train(train_dataloader, validation_dataloader, n_epochs=arguments.epochs)

    return float(load_csv_file(os.path.join(BASE_RESULTS_DIR, arguments.task, parameters_string, NEURAL_TRAINING_SUMMARY_FILENAME))[1][0])


def main(arguments):
    seed_everything(arguments.seed)

    logger.initLogging(arguments.log_level)
    logging.info("Beginning pre-training.")
    logging.debug("Arguments: %s" % (arguments,))
    logging.info("GPU available: %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("Using device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))

    logging.info("Loading Training Dataset")
    train_dataset = RoadRDataset(VIDEO_PARTITIONS[arguments.task]["TRAIN"], TRAIN_VALIDATION_DATA_PATH, arguments.image_resize,
                                 max_frames=arguments.max_frames, use_transforms=arguments.use_transforms)
    validation_dataset = RoadRDataset(VIDEO_PARTITIONS[arguments.task]["VALID"], TRAIN_VALIDATION_DATA_PATH, arguments.image_resize,
                                      max_frames=arguments.max_frames)

    os.makedirs(os.path.join(BASE_RESULTS_DIR, arguments.task, NEURAL_TRAINED_MODEL_DIR), exist_ok=True)

    config = vars(arguments)
    config["default_parameters"] = DEFAULT_PARAMETERS
    write_json_file(os.path.join(BASE_RESULTS_DIR, arguments.task, NEURAL_TRAINED_MODEL_DIR, "config.json"), config)

    loss = run_setting(arguments, train_dataset, validation_dataset, DEFAULT_PARAMETERS, NEURAL_TRAINED_MODEL_DIR)
    logging.info("Final loss: %f" % (loss,))


def _load_args():
    parser = argparse.ArgumentParser(description="RoadR Pre-Training Network")

    parser.add_argument("--task", dest="task", type=str, choices=["task1", "task2"])
    parser.add_argument("--seed", dest="seed",
                        action="store", type=int, default=SEED,
                        help="Seed for random number generator.")
    parser.add_argument("--log-level", dest="log_level",
                        action="store", type=str, default="INFO",
                        help="Logging level.", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--image-resize", dest="image_resize",
                        action="store", type=float, default=1.0,
                        help="Resize factor for all images.")
    parser.add_argument("--max-frames", dest="max_frames",
                        action="store", type=int, default=0,
                        help="Maximum number of frames to use from each videos. Default is 0, which uses all frames.")
    parser.add_argument("--epochs", dest="epochs",
                        action="store", type=int, default=50,
                        help="Number of epochs to train for.")
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint",
                        action="store_true", default=False,
                        help="Resume training from the most recent checkpoint.")
    parser.add_argument("--use-transforms", dest="use_transforms",
                        action="store_true", default=False,
                        help="Use training transforms.")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main(_load_args())
