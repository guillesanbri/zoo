# zoo - Deep Learning Models and Experiments

A collection of Deep Learning models and experiments.

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

### [Dynamic Tanh (DyT)](models/dyt.py)
A custom normalization layer from ["Transformers without Normalization"](https://arxiv.org/abs/2503.10622) that leverages a parameterized tanh function. This can be used as an alternative to Normalization Layers.

### [Vision Transformer (ViT)](models/vit.py)
A minimal implementation of the Vision Transformer architecture as described in the ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) paper.

### [Unet3D](models/unet3d.py)
U-Net based model using 3D convolutions for regression on spatiotemporal data. Follows the implementation proposed in ["3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"](https://arxiv.org/abs/1606.06650)

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

This project is licensed under the MIT License, see the [LICENSE](LICENSE) file for details.
