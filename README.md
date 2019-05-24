# facenet-pytorch

This is a repository for Inception Resnet (V1) models in pytorch, pretrained on VGGFace2 and CASIA-Webface.

Pytorch Models weights were initialized using parameters ported from David Sandberg's [tensorflow facenet repo](https://github.com/davidsandberg/facenet).

## Pretrained models

See: [models/inception_resnet_v1.py](models/inception_resnet_v1.py)

The following models have been ported to pytorch (with links to download pytorch `state_dict`'s):

|Model name|LFW accuracy|Training dataset|
|-|-|-|
|[20180408-102900](https://drive.google.com/uc?export=download&id=1eyCFwxr-0-hyEOdlicD2GISZEWjkWWoa)|0.9905|CASIA-WebFace|
|[20180402-114759](https://drive.google.com/uc?export=download&id=1mN0I8JWpzJ48TrrWFx8jBARwRgk3n6lk)|0.9965|VGGFace2|

To use an Inception Resnet (V1) model for facial recognition/detection, use:

```
from models.inception_resnet_v1 import InceptionResNetV1

# For a model pretrained on VGGFace2
model = InceptionResNetV1(pretrained='vggface2')

# For a model pretrained on CASIA-Webface
model = InceptionResNetV1(pretrained='casia-webface')

# For an untrained model
model = InceptionResNetV1()
```

By default, the above models will return 512-dimensional embeddings of images. To enable classification, either pass `classify=True` to the model constructor, or you can set the object attribute afterwards with `model.classify = True`.

## Conversion of parameters from Tensorflow to Pytorch

See: [models/tensorflow2pytorch.py](models/tensorflow2pytorch.py)

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

## To-do

- Implement dataset, data loader, and image preprocessing for easy prediction.
