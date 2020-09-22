from setuptools import setup, find_packages

import imgcl

with open("README.md", 'r') as f:
    long_description = f.read()

requirements = [
    "torch==1.4.0",
    # "torchvision=0.7.0",
    "tqdm==4.48.2",
    "pandas==1.1.0",
    "wandb==0.10.2"
]

setup(
    name='imgcl',
    version=imgcl.__version__,
    description='Image classification',
    license="MIT",
    long_description=long_description,
    author='Pavel Fakanov',
    author_email='pavel.fakanov@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
)
