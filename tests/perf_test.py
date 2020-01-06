from facenet_pytorch import MTCNN, training
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import time


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on device "{device}"')

    mtcnn = MTCNN(device=device)

    batch_size = 32

    # Generate data loader
    ds = datasets.ImageFolder(
        root='data/test_images/',
        transform=transforms.Resize((512, 512))
    )
    dl = DataLoader(
        dataset=ds,
        num_workers=4,
        collate_fn=training.collate_pil,
        batch_size=batch_size,
        sampler=RandomSampler(ds, replacement=True, num_samples=960),
    )

    start = time.time()
    faces = []
    for x, _ in tqdm(dl):
        faces.extend(mtcnn(x))
    elapsed = time.time() - start
    print(f'Elapsed: {elapsed} | EPS: {len(dl) * batch_size / elapsed}')


if __name__ == '__main__':
    main()
