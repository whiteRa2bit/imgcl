from imgcl.config import TRAIN_VAL_DIR, TRAIN_VAL_LABELS_PATH
from imgcl.dataset import ImageDataset


def debug_model(model):
    dataset = ImageDataset(TRAIN_VAL_DIR, TRAIN_VAL_LABELS_PATH)
    test_img = dataset[0]['image'].unsqueeze(0)
    model(test_img, debug=True)
