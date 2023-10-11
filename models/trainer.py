import os
import time

import torch
import tqdm

from torch.utils.data import DataLoader

from models.losses import detr_loss
from utils import save_model_state
from utils import NEURAL_TRAINED_MODEL_FILENAME
from utils import NEURAL_TRAINING_CONVERGENCE_FILENAME
from utils import NEURAL_TRAINING_SUMMARY_FILENAME
from utils import NEURAL_VALIDATION_CONVERGENCE_FILENAME
from utils import NEURAL_VALIDATION_SUMMARY_FILENAME


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

                    frame_ids, pixel_values, pixel_mask, labels, boxes = batch
                    self.batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})
                    loss, results = detr_loss(self.batch_predictions["pred_boxes"], self.batch_predictions["logits"], boxes, labels)

                    epoch_loss += loss.item()
                    epoch_bce_loss += results["bce_loss"] * results["bce_weight"]
                    epoch_giou_loss += results["giou_loss"] * results["giou_weight"]
                    epoch_l1_loss += results["l1_loss"] * results["l1_weight"]

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
                    }

                    tq.set_postfix(**postfix_data)

                self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time

            learning_convergence += "{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f} \n".format(
                total_time, epoch_time, epoch_loss / dataset_size,
                epoch_bce_loss / dataset_size, epoch_giou_loss / dataset_size,
                epoch_l1_loss / dataset_size)

            if (epoch % compute_period == 0) or (epoch == n_epochs - 1):
                with open(os.path.join(self.out_directory, NEURAL_TRAINING_CONVERGENCE_FILENAME), "w") as training_convergence_checkpoint_file:
                    training_convergence_checkpoint_file.write("Total Time(s), Epoch Time(s), Total Loss, BCE Loss, GIOU Loss, L1 Loss\n")
                    training_convergence_checkpoint_file.write(learning_convergence)

                save_model_state(self.model, self.out_directory, "epoch_{}_".format(epoch) + NEURAL_TRAINED_MODEL_FILENAME)

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

    def compute_total_loss(self, dataloader: DataLoader):
        """
        Evaluate the model on the provided data.
        :param dataloader: The data to evaluate the model on.
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

                frame_ids, pixel_values, pixel_mask, labels, boxes = batch
                self.batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})
                loss, results = detr_loss(self.batch_predictions["pred_boxes"], self.batch_predictions["logits"], boxes, labels)

                total_loss += loss.item()
                total_bce_loss += results["bce_loss"] * results["bce_weight"]
                total_giou_loss += results["giou_loss"] * results["giou_weight"]
                total_l1_loss += results["l1_loss"] * results["l1_weight"]

                frame_ids, pixel_values, pixel_mask, labels, boxes = batch

                predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})

                all_box_predictions.extend(predictions["pred_boxes"].cpu().tolist())
                all_logits.extend(predictions["logits"].cpu().tolist())
                all_frame_indexes.extend(frame_ids.cpu().tolist())

        total_results = {
            "bce_loss": total_bce_loss / dataset_size,
            "giou_loss": total_giou_loss / dataset_size,
            "l1_loss": total_l1_loss / dataset_size,
            "loss": total_loss / dataset_size
        }

        return all_frame_indexes, all_box_predictions, all_logits, total_results

    def post_gradient_computation(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
