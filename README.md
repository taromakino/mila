This repository contains a minimal codebase for performing reproducible deep learning experiments.

Setup:
* Install `numpy, pytorch, torchvision, gin-config`
* Set `DATA_PATH` environment variable

Train a WideResNet on CIFAR-10:

`python experiment.py config/cifar10.gin results`
