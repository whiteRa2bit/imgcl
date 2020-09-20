import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import tqdm
import wandb

from imgcl.utils import get_checkpoint_path
from imgcl.config import TRAIN_SIZE, WANDB_PROJECT, TRAIN_CONFIG, INFERENCE_BATCH_SIZE

_logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, optimizer, dataset, config=TRAIN_CONFIG):
        self.model = model.to(config['device'])
        self.optimizer = optimizer
        self.config = config

        train_dataloader, val_dataloader = self._get_dataloaders(dataset)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def _initialize_wandb(self, project_name=WANDB_PROJECT):
        wandb.init(config=self.config, project=project_name)
        wandb.watch(self.model)

    def _get_dataloaders(self, dataset):
        train_len = int(len(dataset) * TRAIN_SIZE)
        val_len = len(dataset) - train_len
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

        train_dataloader = DataLoader(train_dataset, batch_size=self.config["train_batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config["val_batch_size"], \
                                shuffle=True)

        return train_dataloader, val_dataloader

    def train(self):
        self.model.train()
        self._initialize_wandb()
        criterion = nn.CrossEntropyLoss()
        best_val_accuracy = 0
        for epoch in range(self.config['epochs_num']):
            _logger.info(f"Epoch {epoch} started...")
            for i, data in tqdm.tqdm(enumerate(self.train_dataloader)):
                inputs = data["image"].to(self.config['device'])
                labels = data["label"].to(self.config['device'])

                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = criterion(output, labels)

                # compute gradients
                loss.backward()

                # make a step
                self.optimizer.step()

                loss = loss.item()

                if i % self.config['log_each'] == 0:
                    val_data = next(iter(self.val_dataloader))
                    val_inputs = val_data["image"].to(self.config['device'])
                    val_labels = val_data["label"].to(self.config['device'])
                    val_output = self.model(val_inputs)

                    val_loss = criterion(val_output, val_labels)
                    val_preds = torch.argmax(val_output, axis=1)
                    val_accuracy = torch.sum(val_preds == val_labels).cpu().numpy() / len(val_labels)

                    wandb.log({
                        "Train Loss": loss, \
                        "Val Loss": val_loss, \
                        "Val accuracy": val_accuracy
                    })

                    if val_accuracy > best_val_accuracy:
                        self.save_checkpoint()
                        best_val_accuracy = val_accuracy
        _logger.info(f"Training finished. Best validation accuraccy: {best_val_accuracy}")

    def save_checkpoint(self):
        checkpoint_path = get_checkpoint_path(self.model, self.config)
        torch.save(self.model.state_dict(), checkpoint_path)
