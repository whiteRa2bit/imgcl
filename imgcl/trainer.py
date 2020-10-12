import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import tqdm
import wandb
import matplotlib.pyplot as plt

from imgcl.utils import get_checkpoint_path
from imgcl.config import TRAIN_SIZE, WANDB_PROJECT, TRAIN_CONFIG

_logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, optimizer, dataset, config=TRAIN_CONFIG):
        self.model = model.to(config['device'])
        self.optimizer = optimizer
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.dataset = dataset

        train_dataloader, val_dataloader = self._get_dataloaders()  # TODO: (@whiteRa2bit, 2020-09-20) Pass dataloader to train
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def _initialize_wandb(self, project_name=WANDB_PROJECT):
        wandb.init(config=self.config, project=project_name)
        wandb.watch(self.model)

    def _get_dataloaders(self):
        train_len = int(len(self.dataset) * TRAIN_SIZE)
        val_len = len(self.dataset) - train_len
        train_dataset, val_dataset = random_split(self.dataset, [train_len, val_len])
        val_dataset.is_train = False

        train_dataloader = DataLoader(train_dataset, batch_size=self.config["train_batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config["val_batch_size"], shuffle=True)

        return train_dataloader, val_dataloader

    def train(self):
        self.model.train()
        self._initialize_wandb()

        best_val_accuracy = 0

        for epoch in range(self.config['epochs_num']):
            _logger.info(f"Epoch {epoch} started...")
            for i, data in tqdm.tqdm(enumerate(self.train_dataloader)):
                inputs = data["image"].to(self.config['device'])
                labels = data["label"].to(self.config['device'])

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # compute gradients
                loss.backward()
                # make a step
                self.optimizer.step()

                loss = loss.item()

                if i % self.config['log_each'] == 0:
                    self.optimizer.param_groups[0]['lr'] *= 0.985
                    val_metrics = self._compute_metrics(self.val_dataloader)
                    val_loss = val_metrics['loss']
                    val_sample = val_metrics['sample']
                    val_accuracy = val_metrics['accuracy']

                    train_sample = inputs[0].cpu().data
                    train_sample = train_sample.permute(1, 2, 0)

                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    ax[0].set_title("Train")
                    ax[1].set_title("Val")
                    ax[0].imshow(train_sample)
                    ax[1].imshow(val_sample)
                    wandb.log({
                        "Train Loss": loss, \
                        "Val Loss": val_loss, \
                        "Val accuracy": val_accuracy, \
                        "Images": fig, \
                        "Learning rate": self.optimizer.param_groups[0]['lr']
                    })
                    plt.clf()

                    if val_accuracy > best_val_accuracy:
                        self._save_checkpoint()
                        best_val_accuracy = val_accuracy

        _logger.info(f"Training finished. Best validation accuraccy: {best_val_accuracy}")

    def _compute_metrics(self, dataloader):
        labels = []
        outputs = []
        self.model.eval()
        self.dataset.is_train = False

        for data in dataloader:
            batch_inputs = data["image"].to(self.config['device'])
            batch_labels = data["label"].to(self.config['device'])
            with torch.no_grad():
                batch_outputs = self.model(batch_inputs)
            labels.append(batch_labels)
            outputs.append(batch_outputs)

        labels = torch.cat(labels)
        outputs = torch.cat(outputs)

        loss = self.criterion(outputs, labels).item()
        preds = torch.argmax(outputs, axis=1)
        accuracy = torch.sum(preds == labels).cpu().numpy() / len(labels)

        self.model.train()
        self.dataset.is_train = True

        val_sample = batch_inputs[0].cpu().data
        val_sample = val_sample.permute(1, 2, 0)
        # val_sample = val_sample.int()

        return {"loss": loss, "accuracy": accuracy, "sample": val_sample}

    def _save_checkpoint(self):
        checkpoint_path = get_checkpoint_path(self.model, self.config)
        torch.save(self.model.state_dict(), checkpoint_path)
