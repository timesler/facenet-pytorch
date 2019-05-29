import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd

from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

# Define MTCNN module
# Default params shown for illustration, but not needed
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True
)

# Define Inception Resnet V1 module
# Set classify=True for pretrained classifier
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Define a dataset and data loader
trans = transforms.Compose([
    transforms.Resize(512),
    np.int_,
    transforms.ToTensor(),
    torch.Tensor.byte
])
dataset = datasets.ImageFolder('data/test_images', transform=trans)
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset)

# Perfom MTCNN facial detection
aligned = []
names = []
for x, y in loader:
    x_aligned = mtcnn(x[0])
    aligned.append(x_aligned)
    names.append(dataset.idx_to_class[y[0].item()])

# Calculate image embeddings
aligned = torch.stack(aligned)
embeddings = resnet(aligned)

# Print distance matrix for classes
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names))
