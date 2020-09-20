import pandas as pd

from imgcl.config import TEST_DIR, TEST_LABELS_PATH, TRAIN_CONFIG, ID_COLUMN, LABEL_COLUMN, TEST_LABELS_PATH
from imgcl.dataset import ImageDataset
from imgcl.nets.alexnet import Model
from imgcl.predictor import Predictor


def _transform_preds(pred, max_len=4):
    pred = str(pred)
    pred = '0' * (max_len - len(pred)) + pred
    return pred


def main():
    test_dataset = ImageDataset(TEST_DIR)
    model = Model(TRAIN_CONFIG)
    predictor = Predictor(model, TRAIN_CONFIG)
    preds, idxs = predictor.predict(test_dataset)

    preds = list(map(_transform_preds, preds))
    preds_df = pd.DataFrame({ID_COLUMN: idxs, LABEL_COLUMN: preds})
    preds_df.to_csv(TEST_LABELS_PATH, index=False)


if __name__ == '__main__':
    main()
