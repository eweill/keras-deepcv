# Keras Deep Computer Vision

This repository contains model definitions, training scripts, and other examples for Keras (Tensorflow backend) implementations for classification, detection, and segmentation (computer vision).

## Models

### Classification

- [x] LeNet [Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) [Model](models/classification/lenet.py)
- [ ] AlexNet [Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) [Model](models/classification/alexnet.py)
- [ ] VGG16 and VGG19 [Paper](https://arxiv.org/pdf/1409.1556.pdf)
- [ ] ResNet [Paper](https://arxiv.org/pdf/1512.03385v1.pdf)
- [ ] YOLO9000 [Paper](https://arxiv.org/pdf/1612.08242.pdf)
- [ ] DenseNet [Paper](https://arxiv.org/pdf/1608.06993.pdf)

### Detection
- [ ] Faster RCNN [Paper](https://arxiv.org/pdf/1506.01497.pdf)
- [ ] SSD [Paper](https://arxiv.org/pdf/1512.02325)
- [ ] YOLOv2 [Paper](https://arxiv.org/pdf/1612.08242.pdf)
- [ ] R-FCN [Paper](https://arxiv.org/pdf/1605.06409.pdf)

### Segmentation
- [ ] FCN8 [Paper](https://arxiv.org/pdf/1411.4038.pdf)
- [ ] SegNet [Paper](https://arxiv.org/pdf/1511.00561)
- [ ] U-Net [Paper](https://arxiv.org/pdf/1505.04597)
- [ ] E-Net [Paper](https://arxiv.org/pdf/1606.02147.pdf)
- [ ] ResNetFCN [Paper](https://arxiv.org/pdf/1611.10080.pdf)
- [ ] PSPNet [Paper](https://arxiv.org/pdf/1612.01105.pdf)
- [ ] Mask RCNN [Paper](https://arxiv.org/pdf/1703.06870.pdf)

## Datasets

### Classification

- [ ] MNIST
- [ ] CIFAR-10/CIFAR-100
- [ ] ImageNet
- [ ] Pascal VOC

### Detection
- [ ] Pascal VOC
- [ ] LISA Traffic Sign
- [ ] KITTI
- [ ] MSCOCO

### Segmentation
- [ ] CamVid
- [ ] Cityscapes
- [ ] Pascal VOC
- [ ] KITTI
- [ ] SYNTHIA
- [ ] GTA-V Segmentation
- [ ] MSCOCO

## Prerequisites

For the models in thie repo, [Keras](https://github.com/fchollet/keras) and [Tensorflow](https://github.com/tensorflow/tensorflow) are required.  Make sure the latest versions are installed.

After these packages have been installed, a few other packages are required (which can be found in requirements.txt)

	pip install -r requirements.txt

## Acknowledgments

This repo would like to acknowledge the following pieces of code which played a part in the development of this repository:

- [keras_zoo](https://github.com/david-vazquez/keras_zoo.git)