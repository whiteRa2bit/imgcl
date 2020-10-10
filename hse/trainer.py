import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils import get_checkpoint_path
from config import TRAIN_SIZE, TRAIN_CONFIG


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

        best_val_accuracy = 0
        for epoch in range(self.config['epochs_num']):
            print(f"Epoch {epoch} started...")
            for i, data in enumerate(self.train_dataloader):
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
                    val_metrics = self._compute_metrics(self.val_dataloader)
                    val_loss = val_metrics['loss']
                    val_accuracy = val_metrics['accuracy']

                    print(f"Val loss: {round(val_loss, 5)}, Val accuracy: {round(val_accuracy, 5)}")

                    if val_accuracy > best_val_accuracy:
                        self._save_checkpoint()
                        best_val_accuracy = val_accuracy
        print("Best val accuracy:", best_val_accuracy)

    def _compute_metrics(self, dataloader):
        self.model.eval()

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

        self.model.train()

        return {"loss": loss, "accuracy": accuracy}

    def _save_checkpoint(self):
        checkpoint_path = get_checkpoint_path(self.model, self.config)
        torch.save(self.model.state_dict(), checkpoint_path)
