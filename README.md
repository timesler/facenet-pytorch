# Face Recognition Using Pytorch 
[![Code Coverage](https://img.shields.io/codecov/c/github/timesler/facenet-pytorch.svg)](https://codecov.io/gh/timesler/facenet-pytorch)

| System | 3.5 | 3.6 | 3.7 |
| :---: | :---: | :---: | :---: |
| Linux | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) |
| macOS | - | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) |

This is a repository for Inception Resnet (V1) models in pytorch, pretrained on VGGFace2 and CASIA-Webface.

Pytorch model weights were initialized using parameters ported from David Sandberg's [tensorflow facenet repo](https://github.com/davidsandberg/facenet).

Included in this repo is an efficient pytorch implementation of MTCNN for face detection prior to inference with Inception Resnet models. Theses models are also pretrained. 

## Pretrained models

See: [models/inception_resnet_v1.py](models/inception_resnet_v1.py)

The following models have been ported to pytorch (with links to download pytorch `state_dict`'s):

|Model name|LFW accuracy (listed [here](https://github.com/davidsandberg/facenet))|Training dataset|
|-|-|-|
|[20180408-102900](https://drive.google.com/uc?export=download&id=12DYdlLesBl3Kk51EtJsyPS8qA7fErWDX) (111MB)|0.9905|CASIA-Webface|
|[20180402-114759](https://drive.google.com/uc?export=download&id=1TDZVEBudGaEd5POR5X4ZsMvdsh1h68T1) (107MB)|0.9965|VGGFace2|

There is no need to manually download the pretrained `state_dict`'s; they are downloaded automatically on model instantiation. To use an Inception Resnet (V1) model for facial recognition/identification in pytorch, use:

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

By default, the above models will return 512-dimensional embeddings of images. To enable classification instead, either pass `classify=True` to the model constructor, or you can set the object attribute afterwards with `model.classify = True`. For VGGFace2, the pretrained model will output probability vectors of length 8631, and for CASIA-Webface probability vectors of length 10575.

## Complete detection and recognition pipeline

Face recognition can be easily applied to raw images by first detecting faces using MTCNN before calculating embedding or probabilities using an Inception Resnet model. 

Note that for real-world datasets, the below code should be modified to control batch sizes being passed to the Resnet. 

See [models/utils/example.py](models/utils/example.py). This can be run with `python -c "import models.utils.example"`.

Output:
```bash
                angelina_jolie  bradley_cooper  kate_siegel  paul_rudd  shea_whigham
angelina_jolie        0.000000        1.389796     0.795807   1.461954      1.455719
bradley_cooper        1.389796        0.000000     1.270406   0.896578      0.906192
kate_siegel           0.795807        1.270406     0.000000   1.321325      1.423997
paul_rudd             1.461954        0.896578     1.321325   0.000000      1.087301
shea_whigham          1.455719        0.906192     1.423997   1.087301      0.000000
```

## Use this repo in your own project

To use pretrained MTCNN and Inception Resnet V1 models in your own project, I recommend first adding this repo as a submodule. Note that the dash ('-') in the repo name should be removed when cloning as a submodule as it will break python when importing:

`git submodule add https://github.com/timesler/facenet-pytorch.git facenet_pytorch`

Models can then be instantiated simply with the following:

```python
import facenet_pytorch as fp

mtcnn = fp.MTCNN()
resnet = fp.InceptionResnetV1(pretrained='vggface2').eval()
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

Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. _VGGFace2: A dataset for recognising face across pose and age_, International Conference on Automatic Face and Gesture Recognition, 2018. [PDF](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)

D. Yi, Z. Lei, S. Liao and S. Z. Li. _CASIAWebface: Learning Face Representation from Scratch_, arXiv:1411.7923v1, 2014. [PDF](https://arxiv.org/pdf/1411.7923)

K. Zhang, Z. Zhang, Z. Li and Y. Qiao. _Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks_, IEEE Signal Processing Letters, 2016. [PDF](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)

## Todo

* Add CI test for windows
