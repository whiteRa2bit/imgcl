import os

# Data params
DATA_DIR = "/home/pafakanov/data/other/dl"
TRAIN_DIR = os.path.join(DATA_DIR, 'trainval/')
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'labels_trainval.csv')
VAL_DIR = os.path.join(DATA_DIR, 'val/')
VAL_LABELS_PATH = os.path.join(DATA_DIR, 'labels_val.csv')
TEST_DIR = os.path.join(DATA_DIR, 'test/')
TEST_LABELS_PATH = os.path.join(DATA_DIR, 'labels_test.csv')

IDX_SIZE = 5
ID_COLUMN = 'Id'
LABEL_COLUMN = 'Category'

# Training params
TRAIN_SIZE = 0.95

# Model params
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
