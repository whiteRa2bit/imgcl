import os

# Data params
DATA_DIR = "./simple_image_classification"
TRAIN_VAL_DIR = os.path.join(DATA_DIR, "trainval/")
TRAIN_VAL_LABELS_PATH = os.path.join(DATA_DIR, "labels_trainval.csv")
TEST_DIR = os.path.join(DATA_DIR, "test/")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "./labels_test.csv")

IDX_SIZE = 5
ID_COLUMN = "Id"
LABEL_COLUMN = "Category"

# Model params
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

# Training params
RANDOM_SEED = 42
TRAIN_SIZE = 0.95
TRAIN_CONFIG = {
    "lr": 3e-4,
    "epochs_num": 10,
    "log_each": 25,
    "device": "cuda",
    "train_batch_size": 128,
    "val_batch_size": 256,
    "dropout": 0
}

# Inference params
INFERENCE_BATCH_SIZE = 128
