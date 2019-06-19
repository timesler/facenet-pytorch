from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from time import time
import sys, os
import glob

from models.mtcnn import MTCNN, prewhiten
from models.inception_resnet_v1 import InceptionResnetV1, get_torch_home
from models.utils.detect_face import extract_face

checkpoints = glob.glob(os.path.join(get_torch_home(), 'checkpoints/*'))
for c in checkpoints:
    print('Removing {}'.format(c))
    os.remove(c)

crop_files = glob.glob('data/test_images_aligned/**/*.png')
for c in crop_files:
    print('Removing {}'.format(c))
    os.remove(c)


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

dataset = datasets.ImageFolder('data/test_images', transform=trans)
dataset.idx_to_class = {k: v for v, k in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=lambda x: x[0])

mtcnn_pt = MTCNN(device=torch.device('cpu'))

names = []
aligned = []
aligned_fromfile = []
for img, idx in loader:
    name = dataset.idx_to_class[idx]
    start = time()
    img_align = mtcnn_pt(img, save_path='data/test_images_aligned/{}/1.png'.format(name))
    print('MTCNN time: {:6f} seconds'.format(time() - start))

    if img_align is not None:
        names.append(name)
        aligned.append(img_align)
        aligned_fromfile.append(get_image('data/test_images_aligned/{}/1.png'.format(name), trans_cropped))

aligned = torch.stack(aligned)
aligned_fromfile = torch.stack(aligned_fromfile)

# EMBEDDING TESTS

expected = [
    [
        [0.000000, 1.392167, 0.777482, 1.422187, 1.448250],
        [1.392167, 0.000000, 1.288665, 0.868175, 0.907774],
        [0.777482, 1.288665, 0.000000, 1.354270, 1.408071],
        [1.422187, 0.868175, 1.354270, 0.000000, 1.071160],
        [1.448250, 0.907774, 1.408071, 1.071160, 0.000000]
    ],
    [
        [0.000000, 1.320149, 0.846368, 1.361708, 1.213877],
        [1.320149, 0.000000, 1.165031, 1.020736, 1.009726],
        [0.846368, 1.165031, 0.000000, 1.318732, 1.246484],
        [1.361708, 1.020736, 1.318732, 0.000000, 1.075795],
        [1.213877, 1.009726, 1.246484, 1.075795, 0.000000]
    ]
]

for i, ds in enumerate(['vggface2', 'casia-webface']):
    resnet_pt = InceptionResnetV1(pretrained=ds).eval()

    start = time()
    embs = resnet_pt(aligned)
    print('\nResnet time: {:6f} seconds\n'.format(time() - start))

    embs_fromfile = resnet_pt(aligned_fromfile)

    dists = [[(emb - e).norm().item() for e in embs] for emb in embs]
    dists_fromfile = [[(emb - e).norm().item() for e in embs_fromfile] for emb in embs_fromfile]

    print('\nOutput:')
    print(pd.DataFrame(dists, columns=names, index=names))
    print('\nOutput (from file):')
    print(pd.DataFrame(dists_fromfile, columns=names, index=names))
    print('\nExpected:')
    print(pd.DataFrame(expected[i], columns=names, index=names))

    total_error = (torch.tensor(dists) - torch.tensor(expected[i])).norm()
    total_error_fromfile = (torch.tensor(dists_fromfile) - torch.tensor(expected[i])).norm()

    print('\nTotal error: {}, {}'.format(total_error, total_error_fromfile))

    if sys.platform != 'win32':
        assert total_error < 1e-4
        assert total_error_fromfile < 1e-4

# CLASSIFICATION TEST

resnet_pt = InceptionResnetV1(pretrained=ds, classify=True).eval()
prob = resnet_pt(aligned)
if sys.platform != 'win32':
    assert prob.mean().detach().item() - 9.4563e-05 < 1e-5

# MULTI-FACE TEST

mtcnn = MTCNN(keep_all=True)
img = Image.open('data/multiface.jpg')
img_detected = Image.open('data/multiface_detected.png')
boxes, probs = mtcnn.detect(img)

draw = ImageDraw.Draw(img)
for i, box in enumerate(boxes):
    draw.rectangle(box.tolist())

mtcnn(img, save_path='data/tmp.png')
tmp_files = glob.glob('data/tmp*')
for f in tmp_files:
    os.remove(f)

# NO-FACE TEST

img = Image.new('RGB', (512, 512))
mtcnn(img)
mtcnn(img, return_prob=True)

# EXAMPLE TEST

print('\nExample code:')

from models.utils import example
