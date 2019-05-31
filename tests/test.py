from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from time import time

from models.mtcnn import MTCNN, prewhiten
from models.inception_resnet_v1 import InceptionResnetV1

trans = transforms.Compose([
    transforms.Resize(512),
    np.int_,
    transforms.ToTensor(),
    torch.Tensor.byte
])

trans_cropped = transforms.Compose([
    np.array,
    torch.tensor,
    torch.Tensor.float,
    transforms.Lambda(lambda x: x.permute(2, 0, 1)),
    prewhiten
])

def get_image(path, trans):
    img = Image.open(path)
    img = trans(img)
    return img


mtcnn_pt = MTCNN()
resnet_pt = InceptionResnetV1(pretrained='vggface2').eval()

names = ['bradley_cooper', 'shea_whigham', 'paul_rudd', 'kate_siegel', 'angelina_jolie']
aligned = []
aligned_fromfile = []
for name in names:
    img = get_image('data/test_images/{}/1.jpg'.format(name), trans)

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
    [0.000000, 0.907774, 0.868175, 1.288665, 1.392167],
    [0.907774, 0.000000, 1.071160, 1.408071, 1.448250],
    [0.868175, 1.071160, 0.000000, 1.354270, 1.422187],
    [1.288665, 1.408071, 1.354270, 0.000000, 0.777482],
    [1.392167, 1.448250, 1.422187, 0.777482, 0.000000]
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
