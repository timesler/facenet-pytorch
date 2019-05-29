import torch
from torch import nn
import torchvision.transforms.functional as F
import numpy as np
from collections import OrderedDict
import os
from PIL import Image
import tensorflow as tf

from dependencies.utils.detect_face import detect_face


class PNet(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../data/pnet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()

        if x.shape[3] == 3:
            x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a).permute(0, 2, 3, 1)
        b = self.conv4_2(x).permute(0, 2, 3, 1)
        return b.numpy(), a.numpy()


class RNet(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../data/rnet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()

        if x.shape[3] == 3:
            x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b.numpy(), a.numpy()


class ONet(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../data/onet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()

        if x.shape[3] == 3:
            x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b.numpy(), c.numpy(), a.numpy()


class MTCNN(nn.Module):

    def __init__(
        self, image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.prewhiten = prewhiten
        
        self.pnet = PNet(True)
        self.rnet = RNet(True)
        self.onet = ONet(True)

    def forward(self, img, save_path=None, return_prob=False):
        # TODO: rewrite this using pytorch tensors and allow passing batches

        with torch.no_grad():
            img = img.permute(1, 2, 0)

            boxes, _ = detect_face(
                img.numpy(), self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor
            )

            if boxes.shape[0] == 0:
                print('Face not found')
                return None, None
    
            prob = torch.tensor(boxes[0, 4])
            box = torch.tensor(boxes[0, 0:4]).squeeze()
            box = torch.tensor([
                (box[0] - self.margin/2).clamp(min=0),
                (box[1] - self.margin/2).clamp(min=0),
                (box[2] + self.margin/2).clamp(max=img.shape[1]),
                (box[3] + self.margin/2).clamp(max=img.shape[0])
            ]).int()

            img = img[box[1]:box[3], box[0]:box[2], :].permute(2, 0, 1)
            img = F.to_pil_image(img)
            img = F.resize(img, (self.image_size, self.image_size))

            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img.save(save_path)

            img = torch.tensor(np.array(img)).float().permute(2, 0, 1)
            
            if self.prewhiten:
                img = prewhiten(img)

            if return_prob:
                return img, prob
            else:
                return img


def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y
