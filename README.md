# Face Recognition Using Pytorch 
[![Downloads](https://pepy.tech/badge/facenet-pytorch)](https://pepy.tech/project/facenet-pytorch)

[![Code Coverage](https://img.shields.io/codecov/c/github/timesler/facenet-pytorch.svg)](https://codecov.io/gh/timesler/facenet-pytorch)

| Python | 3.7 | 3.6 | 3.5 |
| :---: | :---: | :---: | :---: |
| Status | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) |

[![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/timesler/facenet-pytorch)

This is a repository for Inception Resnet (V1) models in pytorch, pretrained on VGGFace2 and CASIA-Webface.

Pytorch model weights were initialized using parameters ported from David Sandberg's [tensorflow facenet repo](https://github.com/davidsandberg/facenet).

Also included in this repo is an efficient pytorch implementation of MTCNN for face detection prior to inference. These models are also pretrained. To our knowledge, this is the fastest MTCNN implementation available.

## Table of contents

* [Table of contents](#table-of-contents)
* [Quick start](#quick-start)
* [Pretrained models](#pretrained-models)
* [Example notebooks](#example-notebooks)
  + [*Complete detection and recognition pipeline*](#complete-detection-and-recognition-pipeline)
  + [*Face tracking in video streams*](#face-tracking-in-video-streams)
  + [*Finetuning pretrained models with new data*](#finetuning-pretrained-models-with-new-data)
  + [*Guide to MTCNN in facenet-pytorch*](#guide-to-mtcnn-in-facenet-pytorch)
  + [*Performance comparison of face detection packages*](#performance-comparison-of-face-detection-packages)
  + [*The FastMTCNN algorithm*](#the-fastmtcnn-algorithm)
* [Running with docker](#running-with-docker)
* [Use this repo in your own git project](#use-this-repo-in-your-own-git-project)
* [Conversion of parameters from Tensorflow to Pytorch](#conversion-of-parameters-from-tensorflow-to-pytorch)
* [References](#references)

## Quick start

1. Install:
    
    ```bash
    # With pip:
    pip install facenet-pytorch
    
    # or clone this repo, removing the '-' to allow python imports:
    git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
    
    # or use a docker container (see https://github.com/timesler/docker-jupyter-dl-gpu):
    docker run -it --rm timesler/jupyter-dl-gpu pip install facenet-pytorch && ipython
    ```
    
1. In python, import facenet-pytorch and instantiate models:
    
    ```python
    from facenet_pytorch import MTCNN, InceptionResnetV1
    
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)
    
    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    ```
    
1. Process an image:
    
    ```python
    from PIL import Image
    
    img = Image.open(<image path>)

    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path=<optional save path>)

    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))

    # Or, if using for VGGFace2 classification
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))
    ```

See `help(MTCNN)` and `help(InceptionResnetV1)` for usage and implementation details.

## Pretrained models

See: [models/inception_resnet_v1.py](models/inception_resnet_v1.py)

The following models have been ported to pytorch (with links to download pytorch state_dict's):

|Model name|LFW accuracy (as listed [here](https://github.com/davidsandberg/facenet))|Training dataset|
| :- | :-: | -: |
|[20180408-102900](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt) (111MB)|0.9905|CASIA-Webface|
|[20180402-114759](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt) (107MB)|0.9965|VGGFace2|

There is no need to manually download the pretrained state_dict's; they are downloaded automatically on model instantiation and cached for future use in the torch cache. To use an Inception Resnet (V1) model for facial recognition/identification in pytorch, use:

```python
from facenet_pytorch import InceptionResnetV1

# For a model pretrained on VGGFace2
model = InceptionResnetV1(pretrained='vggface2').eval()

# For a model pretrained on CASIA-Webface
model = InceptionResnetV1(pretrained='casia-webface').eval()

# For an untrained model with 100 classes
model = InceptionResnetV1(num_classes=100).eval()

# For an untrained 1001-class classifier
model = InceptionResnetV1(classify=True, num_classes=1001).eval()
```

Both pretrained models were trained on 160x160 px images, so will perform best if applied to images resized to this shape. For best results, images should also be cropped to the face using MTCNN (see below).

By default, the above models will return 512-dimensional embeddings of images. To enable classification instead, either pass `classify=True` to the model constructor, or you can set the object attribute afterwards with `model.classify = True`. For VGGFace2, the pretrained model will output logit vectors of length 8631, and for CASIA-Webface logit vectors of length 10575.

## Example notebooks

### *Complete detection and recognition pipeline*

Face recognition can be easily applied to raw images by first detecting faces using MTCNN before calculating embedding or probabilities using an Inception Resnet model. The example code at [examples/infer.ipynb](examples/infer.ipynb) provides a complete example pipeline utilizing datasets, dataloaders, and optional GPU processing.

### *Face tracking in video streams*

MTCNN can be used to build a face tracking system (using the `MTCNN.detect()` method). A full face tracking example can be found at [examples/face_tracking.ipynb](examples/face_tracking.ipynb).

![](examples/tracked.gif)

### *Finetuning pretrained models with new data*

In most situations, the best way to implement face recognition is to use the pretrained models directly, with either a clustering algorithm or a simple distance metrics to determine the identity of a face. However, if finetuning is required (i.e., if you want to select identity based on the model's output logits), an example can be found at [examples/finetune.ipynb](examples/finetune.ipynb).

### *Guide to MTCNN in facenet-pytorch*

This guide demonstrates the functionality of the MTCNN module. Topics covered are:

* Basic usage
* Image normalization
* Face margins
* Multiple faces in a single image
* Batched detection
* Bounding boxes and facial landmarks
* Saving face datasets

See the [notebook on kaggle](https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch).

### *Performance comparison of face detection packages*

This notebook demonstrates the use of three face detection packages:

1. facenet-pytorch
1. mtcnn
1. dlib

Each package is tested for its speed in detecting the faces in a set of 300 images (all frames from one video), with GPU support enabled. Performance is based on Kaggle's P100 notebook kernel. Results are summarized below.

|Package|FPS (1080x1920)|FPS (720x1280)|FPS (540x960)|
|---|---|---|---|
|facenet-pytorch|12.97|20.32|25.50|
|facenet-pytorch (non-batched)|9.75|14.81|19.68|
|dlib|3.80|8.39|14.53|
|mtcnn|3.04|5.70|8.23|

![](examples/performance-comparison.png)

See the [notebook on kaggle](https://www.kaggle.com/timesler/comparison-of-face-detection-packages).

### *The FastMTCNN algorithm*

This algorithm demonstrates how to achieve extremely efficient face detection specifically in videos, by taking advantage of similarities between adjacent frames.

See the [notebook on kaggle](https://www.kaggle.com/timesler/fast-mtcnn-detector-55-fps-at-full-resolution).

## Running with docker

The package and any of the example notebooks can be run with docker (or nvidia-docker) using:

```bash
docker run --rm -p 8888:8888
    -v ./facenet-pytorch:/home/jovyan timesler/jupyter-dl-gpu \
    -v <path to data>:/home/jovyan/data
    pip install facenet-pytorch && jupyter lab 
```

Navigate to the examples/ directory and run any of the ipython notebooks.

See [timesler/jupyter-dl-gpu](https://github.com/timesler/docker-jupyter-dl-gpu) for docker container details.

## Use this repo in your own git project

To use this code in your own git repo, I recommend first adding this repo as a submodule. Note that the dash ('-') in the repo name should be removed when cloning as a submodule as it will break python when importing:

`git submodule add https://github.com/timesler/facenet-pytorch.git facenet_pytorch`

Alternatively, the code can be installed as a package using pip:

`pip install facenet-pytorch`

## Conversion of parameters from Tensorflow to Pytorch

See: [models/utils/tensorflow2pytorch.py](models/tensorflow2pytorch.py)

Note that this functionality is not needed to use the models in this repo, which depend only on the saved pytorch `state_dict`'s. 

Following instantiation of the pytorch model, each layer's weights were loaded from equivalent layers in the pretrained tensorflow models from [davidsandberg/facenet](https://github.com/davidsandberg/facenet).

The equivalence of the outputs from the original tensorflow models and the pytorch-ported models have been tested and are identical:

---

`>>> compare_model_outputs(mdl, sess, torch.randn(5, 160, 160, 3).detach())`

```
Passing test data through TF model

tensor([[-0.0142,  0.0615,  0.0057,  ...,  0.0497,  0.0375, -0.0838],
        [-0.0139,  0.0611,  0.0054,  ...,  0.0472,  0.0343, -0.0850],
        [-0.0238,  0.0619,  0.0124,  ...,  0.0598,  0.0334, -0.0852],
        [-0.0089,  0.0548,  0.0032,  ...,  0.0506,  0.0337, -0.0881],
        [-0.0173,  0.0630, -0.0042,  ...,  0.0487,  0.0295, -0.0791]])

Passing test data through PT model

tensor([[-0.0142,  0.0615,  0.0057,  ...,  0.0497,  0.0375, -0.0838],
        [-0.0139,  0.0611,  0.0054,  ...,  0.0472,  0.0343, -0.0850],
        [-0.0238,  0.0619,  0.0124,  ...,  0.0598,  0.0334, -0.0852],
        [-0.0089,  0.0548,  0.0032,  ...,  0.0506,  0.0337, -0.0881],
        [-0.0173,  0.0630, -0.0042,  ...,  0.0487,  0.0295, -0.0791]],
       grad_fn=<DivBackward0>)

Distance 1.2874517096861382e-06
```

---

In order to re-run the conversion of tensorflow parameters into the pytorch model, ensure you clone this repo _with submodules_, as the davidsandberg/facenet repo is included as a submodule and parts of it are required for the conversion.

## References

1. David Sandberg's facenet repo: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)

1. F. Schroff, D. Kalenichenko, J. Philbin. _FaceNet: A Unified Embedding for Face Recognition and Clustering_, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832)

1. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. _VGGFace2: A dataset for recognising face across pose and age_, International Conference on Automatic Face and Gesture Recognition, 2018. [PDF](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)

1. D. Yi, Z. Lei, S. Liao and S. Z. Li. _CASIAWebface: Learning Face Representation from Scratch_, arXiv:1411.7923, 2014. [PDF](https://arxiv.org/pdf/1411.7923)

1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. _Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks_, IEEE Signal Processing Letters, 2016. [PDF](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)
