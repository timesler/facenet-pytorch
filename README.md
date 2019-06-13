# Face Recognition Using Pytorch 
[![Code Coverage](https://img.shields.io/codecov/c/github/timesler/facenet-pytorch.svg)](https://codecov.io/gh/timesler/facenet-pytorch)

| System | Python | |
| :---: | :---: | :---: |
| Linux | 3.5, 3.6, 3.7 | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) |
| macOS | 3.6, 3.7 | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) |
| Windows | 3.5, 3.6, 3.7 | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) |

This is a repository for Inception Resnet (V1) models in pytorch, pretrained on VGGFace2 and CASIA-Webface.

Pytorch model weights were initialized using parameters ported from David Sandberg's [tensorflow facenet repo](https://github.com/davidsandberg/facenet).

Also included in this repo is an efficient pytorch implementation of MTCNN for face detection prior to inference. These models are also pretrained.

## Quick start

1. Clone this repo, removing the '-' to allow python imports:
    ```git
    git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
    ```
1. In python, import the module:
    ```python
    from facenet_pytorch import MTCNN, InceptionResnetV1
    ```
1. If required, create a face _detection_ pipeline using MTCNN:
    ```python
    mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)
    ```
1. Create an inception resnet (in eval mode):
    ```python
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
|[20180408-102900](https://drive.google.com/uc?export=download&id=12DYdlLesBl3Kk51EtJsyPS8qA7fErWDX) (111MB)|0.9905|CASIA-Webface|
|[20180402-114759](https://drive.google.com/uc?export=download&id=1TDZVEBudGaEd5POR5X4ZsMvdsh1h68T1) (107MB)|0.9965|VGGFace2|

There is no need to manually download the pretrained state_dict's; they are downloaded automatically on model instantiation and cached for future use in the torch cache. To use an Inception Resnet (V1) model for facial recognition/identification in pytorch, use:

```python
from models.inception_resnet_v1 import InceptionResnetV1

# For a model pretrained on VGGFace2
model = InceptionResnetV1(pretrained='vggface2').eval()

# For a model pretrained on CASIA-Webface
model = InceptionResnetV1(pretrained='casia-webface').eval()

# For an untrained model
model = InceptionResnetV1().eval()

# For an untrained 1001-class classifier
model = InceptionResnetV1(classify=True, num_classes=1001).eval()
```

Both pretrained models were trained on 160x160 px images, so will perform best if applied to images resized to this shape. For best results, images should also be cropped to the face using MTCNN (see below).

By default, the above models will return 512-dimensional embeddings of images. To enable classification instead, either pass `classify=True` to the model constructor, or you can set the object attribute afterwards with `model.classify = True`. For VGGFace2, the pretrained model will output probability vectors of length 8631, and for CASIA-Webface probability vectors of length 10575.

## Complete detection and recognition pipeline

Face recognition can be easily applied to raw images by first detecting faces using MTCNN before calculating embedding or probabilities using an Inception Resnet model.

The example code at [models/utils/example.py](models/utils/example.py) provides a complete example pipeline utilizing datasets, dataloaders, and optional GPU processing. From the repo directory, this can be run with `python -c "import models.utils.example"`.

Note that for real-world datasets, code should be modified to control batch sizes being passed to the Resnet, particularly if being processed on a GPU. Furthermore, for repeated testing, it is best to separate face detection (using MTCNN) from embedding or classification (using InceptionResnetV1), as detection can then be performed a single time and detected faces saved for future use.

## Use this repo in your own git project

To use pretrained MTCNN and Inception Resnet V1 models in your own git repo, I recommend first adding this repo as a submodule. Note that the dash ('-') in the repo name should be removed when cloning as a submodule as it will break python when importing:

`git submodule add https://github.com/timesler/facenet-pytorch.git facenet_pytorch`

Models can then be instantiated simply with the following:

```python
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()
```

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
