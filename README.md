# Segmentation Research
Tensorflow Implementation of Dilated Residual Networks and other Segementation Networks

### Backbones Supported
- ResNet - https://arxiv.org/abs/1512.03385
- Dilated ResNet - https://arxiv.org/abs/1705.09914

### Networks Supported

- PSPNet - https://arxiv.org/abs/1612.01105
- FPN (In Progress) - https://arxiv.org/abs/1612.03144

### Acknowledgements

This repository structure inspired by the great [segmentation_models](https://github.com/qubvel/segmentation_models). The reason I created this package rather than using this is to have a finer control over available backbones, auxillary losses, and normalization techniques throughout the network.