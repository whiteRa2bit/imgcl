import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import tqdm
import wandb

from imgcl.config import TRAIN_SIZE, WANDB_PROJECT
from imgcl.training.utils import save_checkpoint

class Trainer:
    def __init__(self, model, optimizer, dataset, config):
        self.model = model.to(config['device'])
        self.optimizer = optimizer
        self.config = config
        
        train_dataloader, val_dataloader = self._get_dataloaders(dataset)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


    def _initialize_wandb(self):
        wandb.init(config=self.config, project=WANDB_PROJECT)
        wandb.watch(self.model)


    def _get_dataloaders(self, dataset):
        train_len = int(len(dataset) * TRAIN_SIZE)
        val_len = len(dataset) - train_len
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

        train_dataloader = DataLoader(train_dataset, batch_size=self.config["train_batch_size"],
                                shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config["val_batch_size"], \
                                shuffle=True)

        return train_dataloader, val_dataloader


    def train(self):
        self.model.train()
        self._initialize_wandb()
        criterion = nn.CrossEntropyLoss()
        best_val_accuracy = 0
        for _ in range(self.config['epochs_num']):
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
                        save_checkpoint(self.model, self.config)
                        best_val_accuracy = val_accuracy
