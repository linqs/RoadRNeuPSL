import os
import time
import torch
import tqdm

from models.losses import binary_cross_entropy
from models.losses import pairwise_generalized_box_iou
from models.hungarian_match import hungarian_match
from torch.utils.data import DataLoader
from typing import Tuple, List

from utils import TRAINING_CONVERGENCE_FILENAME
from utils import TRAINING_SUMMARY_FILENAME
from models.model_utils import save_model_state


class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler, device: torch.device, out_directory: str):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.out_directory = out_directory

    def train(self, training_data: DataLoader, validation_data: DataLoader,
              n_epochs: int = 500, compute_period: int = 1):
        """
        Train the provided model and log training performance.
        :param training_data: The training data to use for training.
        :param validation_data: The validation data to use for validation.
        :param n_epochs: (Optional) The total number of epochs to run training.
        :param compute_period: (Optional) The number of epochs to run between each validation evaluation.
        """
        learning_convergence = ""

        validation_score = -1
        best_validation_score = -1
        best_loss = 0
        loss_value = 0
        total_time = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for epoch in tqdm.tqdm(range(n_epochs), "Training Model", leave=True):
            epoch_start_time = time.time()

            with tqdm.tqdm(training_data) as tq:
                tq.set_description("Epoch:{}".format(epoch))
                for step, batch in enumerate(tq):
                    # Transfer the batch to the GPU.
                    batch = [b.to(self.device) for b in batch]

                    self.optimizer.zero_grad(set_to_none=True)

                    loss = self.compute_training_loss(batch)

                    print("Loss: {}".format(loss.item()))
                    print("Backward pass")
                    loss.backward()
                    print("Post gradient computation")
                    self.post_gradient_computation()
                    loss_value = loss.item()

                    print("Gradient Step")
                    self.optimizer.step()
                    self.scheduler.step()

                    tq.set_postfix(loss=loss_value, validation_score=validation_score)

            if (epoch % compute_period == 0) or (epoch == n_epochs - 1):
                torch.cuda.empty_cache()
                validation_score = self.compute_validation_score(validation_data)

                if best_validation_score <= validation_score:
                    best_validation_score = validation_score
                    best_loss = loss_value
                    save_model_state(self.model, self.out_directory)

            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time

            learning_convergence += "{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f} \n".format(
                total_time, epoch_time, loss.item(), validation_score, best_validation_score)

        if torch.cuda.is_available():
            max_gpu_mem = torch.cuda.max_memory_allocated()
        else:
            max_gpu_mem = -1

        with open(os.path.join(self.out_directory, TRAINING_CONVERGENCE_FILENAME), 'w') as training_convergence_file:
            training_convergence_file.write(
                "Total Time(s), Epoch Time(s), "
                "Training Loss, "
                "Validation Evaluation, Best Validation Evaluation\n"
            )
            training_convergence_file.write(learning_convergence)

        with open(os.path.join(self.out_directory, TRAINING_SUMMARY_FILENAME), 'w') as training_summary_file:
            training_summary_file.write("Training Loss,Validation Evaluation,Total Time(s),Max GPU memory (B)\n")
            training_summary_file.write("{:.5f}, {:.5f}, {:.5f}, {:d}".format(
                best_loss, best_validation_score, total_time, max_gpu_mem))

    def compute_training_loss(self, batch: (Tuple, List), bce_weight: int = 1, giou_weight: int = 2) -> torch.Tensor:
        """
        Compute the training loss for the provided batch.
        :param batch: The batch to compute the training loss for.
        :param bce_weight: The weight to apply to the binary cross entropy loss. Default is 1 from DETR paper.
        :param giou_weight: The weight to apply to the generalized IoU loss. Default is 2 from DETR paper.
        :return: The training loss for the provided batch.
        """
        return self._compute_loss(batch, bce_weight, giou_weight)

    def compute_validation_score(self, validation_data: DataLoader) -> float:
        """
        Compute the validation score for the provided validation data.
        :param validation_data: The validation data to compute the validation score for.
        :return: The validation score for the provided validation data.
        """
        validation_score = 0.0
        for validation_batch in validation_data:
            # Transfer the batch to the GPU.
            validation_batch = [b.to(self.device) for b in validation_batch]

            validation_score += self._compute_loss(validation_batch).item()
        return validation_score

    def _compute_loss(self, data: (Tuple, List), bce_weight: int = 1, giou_weight: int = 2) -> torch.Tensor:
        """
        Compute the loss for the provided data.
        :param data: The batch to compute the training loss for.
        :param bce_weight: The weight to apply to the binary cross entropy loss. Default is 1 from DETR paper.
        :param giou_weight: The weight to apply to the generalized IoU loss. Default is 2 from DETR paper.
        :return: The training loss for the provided batch.
        """
        frames, train_images, labels, boxes = data

        # Compute the predictions for the provided batch.
        print("Computing predictions")
        predictions = self.model(train_images)
        print("Predictions computed")

        # Compute the training loss.
        # For the training loss, we need to first compute the matching between the predictions and the ground truth.
        print("Computing matching")
        matching = hungarian_match(predictions["boxes"], boxes)
        print("Matching computed")

        # Compute the classification loss using the matching.
        print("Computing losses")
        bce_loss = binary_cross_entropy(predictions["class_probabilities"], labels, matching)
        print("BCE loss computed")

        # Compute the bounding box loss using the matching.
        giou_loss = pairwise_generalized_box_iou(predictions["boxes"], boxes, matching)
        print("GIOU loss computed")

        return bce_weight * bce_loss + giou_weight * giou_loss

    def post_gradient_computation(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
