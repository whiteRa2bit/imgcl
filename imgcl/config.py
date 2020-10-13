import os

# Data params
DATA_DIR = "/home/pafakanov/data/other/dl"
TRAIN_VAL_DIR = os.path.join(DATA_DIR, "trainval/")
TRAIN_VAL_LABELS_PATH = os.path.join(DATA_DIR, "labels_trainval.csv")
TEST_DIR = os.path.join(DATA_DIR, "test/")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "labels_test.csv")

IDX_SIZE = 5
ID_COLUMN = "Id"
LABEL_COLUMN = "Category"

# Model params
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

# WANDB params
WANDB_PROJECT = "dl_hse"

# Training params
TRAIN_SIZE = 0.95
TRAIN_CONFIG = {
    "lr": 3e-4,
    "lr_update_each": 50,
    "epochs_num": 8,
    "log_each": 25,
    "device": "cuda",
    "train_batch_size": 64,
    "val_batch_size": 128,
    "dropout": 0,
    "l2_coef": 0.01,
}

# Inference params
INFERENCE_BATCH_SIZE = 128
