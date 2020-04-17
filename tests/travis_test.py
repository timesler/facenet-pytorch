"""
The following code is intended to be run only by travis for continuius intengration and testing
purposes. For implementation examples see notebooks in the examples folder.
"""

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from time import time
import sys, os
import glob

from models.mtcnn import MTCNN, fixed_image_standardization
from models.inception_resnet_v1 import InceptionResnetV1, get_torch_home


#### CLEAR ALL OUTPUT FILES ####

# checkpoints = glob.glob(os.path.join(get_torch_home(), 'checkpoints/*'))
# for c in checkpoints:
#     print('Removing {}'.format(c))
#     os.remove(c)

crop_files = glob.glob('data/test_images_aligned/**/*.png')
for c in crop_files:
    print('Removing {}'.format(c))
    os.remove(c)


#### TEST EXAMPLE IPYNB'S ####

os.system('jupyter nbconvert --to script --stdout examples/infer.ipynb examples/finetune.ipynb > examples/tmptest.py')
os.chdir('examples')
import examples.tmptest
os.chdir('..')


#### TEST MTCNN ####

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
    fixed_image_standardization
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


#### TEST EMBEDDINGS ####

expected = [
    [
        [0.000000, 1.482895, 0.886342, 1.438450, 1.437583],
        [1.482895, 0.000000, 1.345686, 1.029880, 1.061939],
        [0.886342, 1.345686, 0.000000, 1.363125, 1.338803],
        [1.438450, 1.029880, 1.363125, 0.000000, 1.066040],
        [1.437583, 1.061939, 1.338803, 1.066040, 0.000000]
    ],
    [
        [0.000000, 1.430769, 0.992931, 1.414197, 1.329544],
        [1.430769, 0.000000, 1.253911, 1.144899, 1.079755],
        [0.992931, 1.253911, 0.000000, 1.358875, 1.337322],
        [1.414197, 1.144899, 1.358875, 0.000000, 1.204118],
        [1.329544, 1.079755, 1.337322, 1.204118, 0.000000]
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


#### TEST CLASSIFICATION ####

resnet_pt = InceptionResnetV1(pretrained=ds, classify=True).eval()
prob = resnet_pt(aligned)


#### MULTI-FACE TEST ####

mtcnn = MTCNN(keep_all=True)
img = Image.open('data/multiface.jpg')
boxes, probs = mtcnn.detect(img)

draw = ImageDraw.Draw(img)
for i, box in enumerate(boxes):
    draw.rectangle(box.tolist())

mtcnn(img, save_path='data/tmp.png')


#### MTCNN TYPES TEST ####

img = Image.open('data/multiface.jpg')

mtcnn = MTCNN(keep_all=True)
boxes_ref, _ = mtcnn.detect(img)
_ = mtcnn(img)

mtcnn = MTCNN(keep_all=True).double()
boxes_test, _ = mtcnn.detect(img)
_ = mtcnn(img)

box_diff = boxes_ref[np.argsort(boxes_ref[:,1])] - boxes_test[np.argsort(boxes_test[:,1])]
total_error = np.sum(np.abs(box_diff))
print('\nfp64 Total box error: {}'.format(total_error))

assert total_error < 1e-2


# half is not supported on CPUs, only GPUs
if torch.cuda.is_available():

    mtcnn = MTCNN(keep_all=True, device='cuda').half()
    boxes_test, _ = mtcnn.detect(img)
    _ = mtcnn(img)

    box_diff = boxes_ref[np.argsort(boxes_ref[:,1])] - boxes_test[np.argsort(boxes_test[:,1])]
    print('fp16 Total box error: {}'.format(np.sum(np.abs(box_diff))))

    # test new automatic multi precision to compare
    if hasattr(torch.cuda, 'amp'):
        with torch.cuda.amp.autocast():
            mtcnn = MTCNN(keep_all=True, device='cuda')
            boxes_test, _ = mtcnn.detect(img)
            _ = mtcnn(img)

        box_diff = boxes_ref[np.argsort(boxes_ref[:,1])] - boxes_test[np.argsort(boxes_test[:,1])]
        print('AMP total box error: {}'.format(np.sum(np.abs(box_diff))))

            
#### MULTI-IMAGE TEST ####

mtcnn = MTCNN(keep_all=True)
img = [
    Image.open('data/multiface.jpg'),
    Image.open('data/multiface.jpg')
]
batch_boxes, batch_probs = mtcnn.detect(img)

mtcnn(img, save_path=['data/tmp1.png', 'data/tmp1.png'])
tmp_files = glob.glob('data/tmp*')
for f in tmp_files:
    os.remove(f)


#### NO-FACE TEST ####

img = Image.new('RGB', (512, 512))
mtcnn(img)
mtcnn(img, return_prob=True)
