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

# Training params
TRAIN_SIZE = 0.95

# Model params
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

# WANDB params
WANDB_PROJECT = "dl_hse"
