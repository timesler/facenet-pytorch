from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter 	
from torchvision import datasets, transforms
import numpy as np
import os

from tqdm import tqdm

data_dir = '../../datasets/CASIA-WebFace'

#data_dir = '../../datasets/test'

batch_size = 64
epochs = 20
workers = 0 if os.name == 'nt' else 8


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


# mtcnn = MTCNN(
#     image_size=160, margin=0, min_face_size=20,
#     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
#     device=device
# )
#
#

dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))

# dataset.samples = [
#     (p, p.replace(data_dir, data_dir + '_cropped'))
#         for p, _ in dataset.samples
# ]
#
# loader = DataLoader(
#     dataset,
#     num_workers=workers,
#     batch_size=batch_size,
#     collate_fn=training.collate_pil
# )
#
# for i, (x, y) in enumerate(loader):
#     mtcnn(x, save_path=y)
#     print('\rBatch {} of {}'.format(i + 1, len(loader)))
    
# Remove mtcnn to reduce GPU memory usage

# For an untrained model with 100 classes
# model = InceptionResnetV1(num_classes=100).eval()

# For an untrained 1001-class classifier
# model = InceptionResnetV1(classify=True, num_classes=1001).eval()

resnet = InceptionResnetV1(
    classify=True,
    num_classes=len(dataset.class_to_idx),
    pretrained='vggface2'
).to(device)


optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

PATH = "~/work/facenet_pytorch/examples/checkpoints"
os.mkdir(PATH)
torch.save(resnet, f"{PATH}/model_1.pt")

writer.close()
