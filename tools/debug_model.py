from imgcl.nets.alexnet import Model
from imgcl.nets.utils import debug_model
from imgcl.config import TRAIN_CONFIG


def main():
    model = Model(TRAIN_CONFIG)
    debug_model(model)


if __name__ == '__main__':
    main()
