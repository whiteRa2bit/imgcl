from imgcl.nets.alexnet import Model
from imgcl.nets.utils import debug_model


def main():
    model = Model({"dropout": 0.5})
    debug_model(model)


if __name__ == '__main__':
    main()
