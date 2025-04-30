# zoo - Deep Learning Models and Experiments

A collection of Deep Learning models implementations and experiments.

- [Overview](#overview)
- [Models](#models)
- [Experiments](#experiments)
- [Repo Structure](#structure)
- [License](#license)

## Overview

This repository contains:

- Custom implementation of different architectures.
- Training pipelines for different modalities.
- Docker scripts for reproducible experiments.

## Models

### Vision Transformer (ViT)

A minimal implementation of the Vision Transformer architecture as described in the "An Image is Worth 16x16 Words" paper.

### Dynamic Tanh (DyT)

A custom normalization layer that leverages a parameterized tanh function. This can be used as an alternative to Normalization Layers.

## Experiments

[LayerNorm vs Dynamic Tanh (DyT) Normalization in Small Vision Transformers](https://guillesanbri.com/layernorm-tanh/): Includes a study of the **loss and accuracy dynamics when training small ViT models from scratch on tiny-imagenet-200 with different normalization approaches**. It also includes a brief time analysis comparing the performance of RMSNorm, LayerNorm, and DyT.


## Structure

```
zoo/
├── configs/              # Training configuration files
├── data_utils/           # Data loading and augmentation utilities
├── models/               # Model and Layers implementations
├── docker-build.sh       # Script to build Docker container
├── docker-run.sh         # Script to run Docker container
├── Dockerfile            # Docker configuration
├── train_*.py            # Training script(s)
└── utils.py              # Utility functions
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
