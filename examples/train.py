# Face detection and recognition training pipeline
# 
# The following example illustrates how to use the `facenet_pytorch` python package to perform face detection and recogition on an image dataset using an Inception Resnet V1 pretrained on the VGGFace2 dataset.

from facenet_pytorch import MTCNN, InceptionResnetV1, prewhiten, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import multiprocessing as mp
import os

# Define run parameters

data_dir = '/mnt/windows/Users/times/Data/vggface2/test'
batch_size = 48
epochs = 8
workers = 2

# Determine if an nvidia GPU is available

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Define MTCNN module
# Default params shown for illustration, but not needed.
# Note that, since MTCNN is a collection of neural nets and other code, the
# device must be passed in the following way to enable copying of objects when
# needed internally.
# See `help(MTCNN)` for more details.

# mtcnn = MTCNN(
#     image_size=160, margin=0, min_face_size=20,
#     thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
#     device=device
# )


# Perfom MTCNN facial detection
# Iterate through the DataLoader object and obtained cropped faces.

dataset = datasets.ImageFolder(data_dir)
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=lambda x: x[0], num_workers=workers, shuffle=False)

# timer = training.BatchTimer(per_sample=False)
# for i, (x, y) in enumerate(loader):
#     print(f'\rImages processed: {i + 1:8d} of {len(loader):8d} | fps: {timer():.0f}', end='')
#     save_dir = os.path.join(data_dir + '_cropped', dataset.idx_to_class[y])
#     os.makedirs(save_dir, exist_ok=True)
#     filename = f'{len(os.listdir(save_dir)):05n}.png'
#     mtcnn(x, save_path=os.path.join(save_dir, filename))

# del mtcnn

# Define Inception Resnet V1 module
# Set classify=True for classifier.
# See `help(InceptionResnetV1)` for more details.

resnet = InceptionResnetV1(
    classify=True,
    num_classes=len(dataset.class_to_idx)
).to(device)


# Define optimizer, scheduler, dataset, and dataloader

optimizer = optim.Adam(resnet.parameters(), lr=0.005, weight_decay=2e-4)
# optimizer = optim.Adam(resnet.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, [5, 10])

trans_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    np.float32,
    transforms.ToTensor(),
    prewhiten
])
trans_val = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    prewhiten,
])
train_dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans_train)
val_dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans_val)
img_inds = np.arange(len(train_dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    train_dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    val_dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)


# Define loss and evaluation functions

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}


# Train model

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print(f'\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

writer.close()