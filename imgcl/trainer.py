import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import tqdm
import wandb

from imgcl.utils import get_checkpoint_path
from imgcl.config import TRAIN_SIZE, WANDB_PROJECT, TRAIN_CONFIG

_logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, optimizer, dataset, config=TRAIN_CONFIG):
        self.model = model.to(config['device'])
        self.optimizer = optimizer
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        train_dataloader, val_dataloader = self._get_dataloaders(
            dataset)  # TODO: (@whiteRa2bit, 2020-09-20) Pass dataloader to train
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Lr sheduler
        self.lr = config['lr_init']
        self.lr_beta = config['lr_beta']
        updates_num = self.config['epochs_num'] * len(train_dataloader) - 1
        self.mult = (config['lr_final'] / self.lr) ** (1 / updates_num)
        self.avg_loss = 0.
        self.best_loss = 0.
        self.batch_num = 0

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
                # update lr
                res = self._update_lr(loss)
                if res == -1:
                    return
        
                loss = loss.item()

                if i % self.config['log_each'] == 0:
                    val_metrics = self._compute_metrics(self.val_dataloader)
                    val_loss = val_metrics['loss']
                    val_accuracy = val_metrics['accuracy']
                    wandb.log({
                        "Train Loss": loss, \
                        "Val Loss": val_loss, \
                        "Val accuracy": val_accuracy, \
                        "Learning rate": self.lr
                    })

                    if val_accuracy > best_val_accuracy:
                        self._save_checkpoint()
                        best_val_accuracy = val_accuracy
        _logger.info(f"Training finished. Best validation accuraccy: {best_val_accuracy}")

    def _update_lr(self, loss):
        self.batch_num += 1
        #Compute the smoothed loss
        self.avg_loss = self.lr_beta * self.avg_loss + (1 - self.lr_beta) * loss.item()
        self.smoothed_loss = self.avg_loss / (1 - self.lr_beta ** self.batch_num)
        #Stop if the loss is exploding
        if self.batch_num > 1 and self.smoothed_loss > 4 * self.best_loss:
            return -1
        #Record the best loss
        if self.smoothed_loss < self.best_loss or self.batch_num == 1:
            self.best_loss = self.smoothed_loss

        #Update the lr for the next step
        self.lr *= self.mult
        self.optimizer.param_groups[0]['lr'] = self.lr

    def _compute_metrics(self, dataloader):
        labels = []
        outputs = []

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

        return {"loss": loss, "accuracy": accuracy}

    def _save_checkpoint(self):
        checkpoint_path = get_checkpoint_path(self.model, self.config)
        torch.save(self.model.state_dict(), checkpoint_path)
