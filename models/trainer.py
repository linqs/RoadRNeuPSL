import os
import time

import torch
import tqdm

from models.hungarian_match import hungarian_match
from models.losses import binary_cross_entropy_with_logits
from models.losses import pairwise_generalized_box_iou
from models.losses import pairwise_l1_loss
from models.model_utils import save_model_state
from torch.utils.data import DataLoader
from typing import Tuple, List

from utils import NEURAL_VALIDATION_SUMMARY_FILENAME
from utils import NEURAL_TRAINED_MODEL_FILENAME
from utils import NEURAL_TRAINING_CONVERGENCE_FILENAME
from utils import NEURAL_VALIDATION_CONVERGENCE_FILENAME
from utils import NEURAL_TRAINING_SUMMARY_FILENAME

BCE_WEIGHT = 1
GIOU_WEIGHT = 5
L1_WEIGHT = 2


class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler, device: torch.device, out_directory: str):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.out_directory = out_directory
        self.batch_predictions = None

    def train(self, training_dataloader: DataLoader, validation_dataloader: DataLoader, n_epochs: int = 500, compute_period: int = 1):
        """
        Train the provided model and log training performance.
        :param training_dataloader: The training data to use for training.
        :param validation_dataloader: The validation data to use for training.
        :param n_epochs: (Optional) The total number of epochs to run training.
        :param compute_period: (Optional) The number of epochs to run between each model checkpoint.
        """
        learning_convergence = ""
        validation_convergence = ""

        dataset_size = len(training_dataloader) * training_dataloader.batch_size

        total_time = 0
        epoch_loss = 0
        epoch_bce_loss = 0
        epoch_giou_loss = 0
        epoch_l1_loss = 0

        previous_epoch_boxes = None
        previous_epoch_logits = None

        epoch_box_movement = 0.0
        epoch_logit_movement = 0.0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for epoch in tqdm.tqdm(range(n_epochs), "Training Model", leave=True):
            epoch_start_time = time.time()
            epoch_loss = 0
            epoch_bce_loss = 0
            epoch_giou_loss = 0
            epoch_l1_loss = 0

            current_epoch_boxes = []
            current_epoch_logits = []

            with tqdm.tqdm(training_dataloader) as tq:
                tq.set_description("Epoch:{}".format(epoch))
                for step, batch in enumerate(tq):
                    batch = [b.to(self.device) for b in batch]

                    self.optimizer.zero_grad(set_to_none=True)

                    loss, results = self._compute_batch_loss(batch)

                    epoch_loss += loss.item()
                    epoch_bce_loss += results["bce_loss"]
                    epoch_giou_loss += results["giou_loss"]
                    epoch_l1_loss += results["l1_loss"]

                    current_epoch_boxes.extend(self.batch_predictions["pred_boxes"].cpu().tolist())
                    current_epoch_logits.extend(self.batch_predictions["logits"].cpu().tolist())

                    loss.backward()
                    self.post_gradient_computation()

                    self.optimizer.step()

                    postfix_data = {
                        "loss": epoch_loss / ((step + 1) * training_dataloader.batch_size),
                        "loss_bce": epoch_bce_loss / ((step + 1) * training_dataloader.batch_size),
                        "loss_giou": epoch_giou_loss / ((step + 1) * training_dataloader.batch_size),
                        "loss_l1": epoch_l1_loss / ((step + 1) * training_dataloader.batch_size),
                        "logit_movement": epoch_logit_movement,
                        "box_movement": epoch_box_movement,
                    }

                    tq.set_postfix(**postfix_data)

                self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time

            if previous_epoch_boxes is not None:
                epoch_logit_movement = torch.abs(torch.Tensor(previous_epoch_logits) - torch.Tensor(current_epoch_logits)).mean().item()
                epoch_box_movement = torch.abs(torch.Tensor(previous_epoch_boxes) - torch.Tensor(current_epoch_boxes)).mean().item()

            previous_epoch_boxes = [] + current_epoch_boxes
            previous_epoch_logits = [] + current_epoch_logits

            learning_convergence += "{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f} \n".format(
                total_time, epoch_time, epoch_loss / dataset_size,
                epoch_bce_loss / dataset_size, epoch_giou_loss / dataset_size,
                epoch_l1_loss / dataset_size)

            if (epoch % compute_period == 0) or (epoch == n_epochs - 1):
                with open(os.path.join(self.out_directory, NEURAL_TRAINING_CONVERGENCE_FILENAME), "w") as training_convergence_checkpoint_file:
                    training_convergence_checkpoint_file.write("Total Time(s), Epoch Time(s), Total Loss, BCE Loss, GIOU Loss, L1 Loss\n")
                    training_convergence_checkpoint_file.write(learning_convergence)

                save_model_state(self.model, self.out_directory, "epoch_{}_".format(epoch) + NEURAL_TRAINED_MODEL_FILENAME)

                if (epoch == n_epochs - 1):
                    save_model_state(self.model, self.out_directory, NEURAL_TRAINED_MODEL_FILENAME)

                _, _, _, validation_results = self.compute_total_loss(validation_dataloader)
                validation_convergence += "{:5d}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f} \n".format(
                    epoch, total_time, epoch_time, validation_results["loss"],
                    validation_results["bce_loss"], validation_results["giou_loss"],
                    validation_results["l1_loss"])

                with open(os.path.join(self.out_directory, NEURAL_VALIDATION_CONVERGENCE_FILENAME), "w") as validation_convergence_checkpoint_file:
                    validation_convergence_checkpoint_file.write("Epoch, Total Time(s), Epoch Time(s), Total Loss, BCE Loss, GIOU Loss, L1 Loss\n")
                    validation_convergence_checkpoint_file.write(validation_convergence)

        if torch.cuda.is_available():
            max_gpu_mem = torch.cuda.max_memory_allocated()
        else:
            max_gpu_mem = -1

        with open(os.path.join(self.out_directory, NEURAL_TRAINING_SUMMARY_FILENAME), "w") as training_summary_file:
            training_summary_file.write("Total Loss, BCE Loss, GIOU Loss, L1 Loss, Total Time(s), Max GPU memory (B)\n")
            training_summary_file.write("{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:d}".format(
                epoch_loss / dataset_size,
                epoch_bce_loss / dataset_size,
                epoch_giou_loss / dataset_size,
                epoch_l1_loss / dataset_size,
                total_time, max_gpu_mem))

        _, _, _, validation_results = self.compute_total_loss(validation_dataloader)
        with open(os.path.join(self.out_directory, NEURAL_VALIDATION_SUMMARY_FILENAME), "w") as validation_summary_file:
            validation_summary_file.write("Total Loss, BCE Loss, GIOU Loss, L1 Loss, Total Time(s), Max GPU memory (B)\n")
            validation_summary_file.write("{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:d}".format(
                validation_results["loss"],
                validation_results["bce_loss"],
                validation_results["giou_loss"],
                validation_results["l1_loss"],
                total_time, max_gpu_mem))

    def compute_total_loss(self, dataloader: DataLoader, bce_weight: int = BCE_WEIGHT, giou_weight: int = GIOU_WEIGHT, l1_weight: int = L1_WEIGHT):
        """
        Evaluate the model on the provided data.
        :param dataloader: The data to evaluate the model on.
        :param bce_weight: The weight to apply to the binary cross entropy loss.
        :param giou_weight: The weight to apply to the generalized IoU loss.
        :param l1_weight: The weight to apply to the l1 loss.
        :return: The predictions and the losses for the predictions on the provided data.
        """
        self.model.eval()

        dataset_size = len(dataloader) * dataloader.batch_size

        all_box_predictions = []
        all_logits = []
        all_frame_indexes = []
        total_loss = 0
        total_bce_loss = 0
        total_giou_loss = 0
        total_l1_loss = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with tqdm.tqdm(dataloader) as tq:
            for step, batch in enumerate(tq):
                batch = [b.to(self.device) for b in batch]

                loss, results = self._compute_batch_loss(batch, bce_weight=bce_weight, giou_weight=giou_weight, l1_weight=l1_weight)
                total_loss += loss.item()
                total_bce_loss += results["bce_loss"]
                total_giou_loss += results["giou_loss"]
                total_l1_loss += results["l1_loss"]

                frame_ids, pixel_values, pixel_mask, labels, boxes = batch

                predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})

                all_box_predictions.extend(predictions["pred_boxes"].cpu().tolist())
                all_logits.extend(predictions["logits"].cpu().tolist())
                all_frame_indexes.extend(frame_ids.cpu().tolist())

        total_results = {
            "bce_loss": (bce_weight * total_bce_loss) / dataset_size,
            "giou_loss": (giou_weight * total_giou_loss) / dataset_size,
            "l1_loss": (l1_weight * total_l1_loss) / dataset_size,
            "loss": (bce_weight * total_bce_loss + giou_weight * total_giou_loss + l1_weight * total_l1_loss) / dataset_size
        }

        return all_frame_indexes, all_box_predictions, all_logits, total_results

    def _compute_batch_loss(self, data: (Tuple, List), bce_weight: int = BCE_WEIGHT, giou_weight: int = GIOU_WEIGHT, l1_weight: int = L1_WEIGHT) -> torch.Tensor:
        """
        Compute the loss for the provided data.
        :param data: The batch to compute the training loss for.
        :param bce_weight: The weight to apply to the binary cross entropy loss.
        :param giou_weight: The weight to apply to the generalized IoU loss.
        :return: The training loss for the provided batch.
        """
        frame_ids, pixel_values, pixel_mask, labels, boxes = data

        # Compute the predictions for the provided batch.
        self.batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})

        # Compute the training loss.
        # For the training loss, we need to first compute the matching between the predictions and the ground truth.
        matching = hungarian_match(self.batch_predictions["pred_boxes"], boxes)

        # Compute the classification loss using the matching.
        bce_loss = binary_cross_entropy_with_logits(self.batch_predictions["logits"], labels, matching)

        # Compute the bounding box loss using the matching.
        giou_loss = pairwise_generalized_box_iou(self.batch_predictions["pred_boxes"], boxes, matching)

        # Compute the bounding box l2 loss using the matching.
        l1_loss = pairwise_l1_loss(self.batch_predictions["pred_boxes"], boxes, matching)

        results = {
            "bce_loss": (bce_weight * bce_loss).item(),
            "giou_loss": (giou_weight * giou_loss).item(),
            "l1_loss": (l1_weight * l1_loss).item(),
            "loss": (bce_weight * bce_loss + giou_weight * giou_loss + l1_weight * l1_loss).item()
        }

        return bce_weight * bce_loss + giou_weight * giou_loss + l1_weight * l1_loss, results

    def post_gradient_computation(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
