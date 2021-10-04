# Segmentation Research
Tensorflow Implementation of Dilated Residual Networks and other Segementation Networks

### Backbones Supported
- ResNet - https://arxiv.org/abs/1512.03385
- Dilated Residual Networks (DRN) - https://arxiv.org/abs/1705.09914
- ResNetv2 Dilated Style (alteration of original resnet, differs from above)

### Networks Supported

- PSPNet - https://arxiv.org/abs/1612.01105
- FPN (Partially) - https://arxiv.org/abs/1612.03144

### Acknowledgements

This repository structure inspired by [segmentation_models](https://github.com/qubvel/segmentation_models). 

The reason I created this package rather than using this is to have a finer control over available backbones, auxillary losses, and normalization techniques throughout the network.

The PSPNet implementation also differs, as I am following the model of the original authors more recent repo [semseg](https://github.com/hszhao/semseg).