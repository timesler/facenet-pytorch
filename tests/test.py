from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from time import time

from models.mtcnn import MTCNN, prewhiten
from models.inception_resnet_v1 import InceptionResnetV1


def get_image(path, trans):
    img = Image.open(path)
    img = trans(img)
    return img


trans = transforms.Compose([
    transforms.Resize(512)
])

trans_cropped = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    prewhiten
])

mtcnn_pt = MTCNN()
resnet_pt = InceptionResnetV1(pretrained='vggface2').eval()

dataset = datasets.ImageFolder('data/test_images', transform=trans)
dataset.idx_to_class = {k: v for v, k in dataset.class_to_idx.items()}
loader = DataLoader(dataset, num_workers=8, collate_fn=lambda x: x[0])

names = []
aligned = []
aligned_fromfile = []
for img, idx in loader:
    name = dataset.idx_to_class[idx]
    names.append(name)
    start = time()
    img_align = mtcnn_pt(img, save_path='data/test_images_aligned/{}/1.png'.format(name))
    print('MTCNN time: {:6f} seconds'.format(time() - start))
    aligned.append(img_align)
    
    aligned_fromfile.append(get_image('data/test_images_aligned/{}/1.png'.format(name), trans_cropped))

aligned = torch.stack(aligned)
aligned_fromfile = torch.stack(aligned_fromfile)

start = time()
embs = resnet_pt(aligned)
print('\nResnet time: {:6f} seconds\n'.format(time() - start))

embs_fromfile = resnet_pt(aligned_fromfile)

dists = [[(emb - e).norm().item() for e in embs] for emb in embs]
dists_fromfile = [[(emb - e).norm().item() for e in embs_fromfile] for emb in embs_fromfile]
expected = [
    [0.000000, 1.392167, 0.777482, 1.422187, 1.448250],
    [1.392167, 0.000000, 1.288665, 0.868175, 0.907774],
    [0.777482, 1.288665, 0.000000, 1.354270, 1.408071],
    [1.422187, 0.868175, 1.354270, 0.000000, 1.071160],
    [1.448250, 0.907774, 1.408071, 1.071160, 0.000000]
]

print('\nOutput:')
print(pd.DataFrame(dists, columns=names, index=names))
print('\nOutput (from file):')
print(pd.DataFrame(dists_fromfile, columns=names, index=names))
print('\nExpected:')
print(pd.DataFrame(expected, columns=names, index=names))

total_error = (torch.tensor(dists) - torch.tensor(expected)).norm()
total_error_fromfile = (torch.tensor(dists_fromfile) - torch.tensor(expected)).norm()
if total_error > 1e-4 or total_error_fromfile > 1e-4:
    raise Exception('Difference between output and expected is too large: {}'.format(max(total_error, total_error_fromfile)))
