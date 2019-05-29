from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
from time import time

from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

trans = transforms.Compose([
    transforms.Resize(512),
    np.int_,
    transforms.ToTensor(),
    torch.Tensor.byte
])

def get_image(path, trans):
    img = Image.open(path)
    img = trans(img)
    return img


mtcnn_pt = MTCNN()
resnet_pt = InceptionResnetV1(pretrained='vggface2').eval()

names = ['bradley_cooper', 'shea_whigham', 'paul_rudd', 'kate_siegel', 'angelina_jolie']
aligned = []
for name in names:
    path = f'data/test_images/{name}/1'
    img = get_image(f'data/test_images/{name}/1.jpg', trans)

    start = time()
    img_align = mtcnn_pt(img, save_path=f'data/test_images_aligned/{name}/1.jpg')
    print(f'MTCNN time: {time() - start:6f} seconds')
    aligned.append(img_align)

aligned = torch.stack(aligned)

start = time()
embs = resnet_pt(aligned)
print(f'\nResnet time: {time() - start:6f} seconds\n')

dists = [[(emb - e).norm().item() for e in embs] for emb in embs]
print(pd.DataFrame(dists, columns=names, index=names))
