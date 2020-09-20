import torch
from torch.utils.data import DataLoader
import tqdm

from imgcl.utils import get_checkpoint_path
from imgcl.config import TRAIN_CONFIG, INFERENCE_BATCH_SIZE


class Predictor:
    def __init__(self, model, config=TRAIN_CONFIG):
        self.model = model.to(config['device'])
        self.config = config
        self.load_checkpoint()

    def predict(self, dataset, config=TRAIN_CONFIG):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False)

        preds = []
        idxs = []
        for data in tqdm.tqdm(dataloader):
            inputs = data["image"].to(self.config['device'])
            output = self.model(inputs)
            output_labels = torch.argmax(output, axis=1)
            preds.append(output_labels)
            idxs += data['idx']
        preds = torch.cat(preds).cpu().numpy()
        return preds, idxs

    def load_checkpoint(self):
        checkpoint_path = get_checkpoint_path(self.model, self.config)
        self.model.load_state_dict(torch.load(checkpoint_path))
