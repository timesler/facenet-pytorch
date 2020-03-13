import configparser
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from models.utils import training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
from apex import amp
import time
import logging

if __name__ == '__main__':

    start_time = time.time()

    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    CONFIG_PATH = '../models/utils/cfg.txt'
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    data_dir = config.get("Parameters", "data_dir")
    batch_size = int(config.get("Parameters", "batch_size"))
    epochs = int(config.get("Parameters", "epochs"))
    training_name = config.get("Parameters", "training_name")
    opt = int(config.get("Parameters", "opt_enabled"))
    opt_level = config.get("Parameters", "opt_level")
    cropping = int(config.get("Parameters", "cropping"))

    logger.info('\n------------ Options -------------\n' +
                f"data-dir: {data_dir}\n" +
                f"batch_size: {batch_size}\n" +
                f"epochs: {epochs}\n" +
                f"training_name: {training_name}\n" +
                f"Opt: {opt}\n" +
                f"Opt_level: {opt_level}\n" +
                f"Cropping: {cropping}\n" +
                '-------------- End ---------------')

    workers = 0 if os.name == 'nt' else 8

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))

    # Cropping
    if cropping:
        logger.info("Cropping")
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )

        dataset.samples = [
            (p, p.replace(data_dir, data_dir + '_cropped'))
            for p, _ in dataset.samples
        ]

        loader = DataLoader(
            dataset,
            num_workers=workers,
            batch_size=batch_size,
            collate_fn=training.collate_pil
        )

        for i, (x, y) in enumerate(loader):
            mtcnn(x, save_path=y)
            print('Batch {} of {}'.format(i + 1, len(loader)))

    resnet = InceptionResnetV1(
        classify=True,
        num_classes=len(dataset.class_to_idx),
        pretrained='vggface2'
    ).to(device)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)

    if opt:
        model, optimizer = amp.initialize(resnet, optimizer, opt_level=opt_level)

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

    logger.info('Initial')
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer, opt=opt
    )

    print()
    logger.info("Start training")

    for epoch in range(epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, epochs))

        resnet.train()
        logger.info("Training")
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer, opt=opt
        )

        logger.info("Validating")
        resnet.eval()
        training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer, opt=opt
        )
        print()

    # saving model's checkpoint
    if opt:
        checkpoint = {
            'model': resnet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict()
        }
        torch.save(checkpoint, f"checkpoints/opt_{training_name}.pt")
    else:
        checkpoint = {
            'model': resnet.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/{training_name}.pt")

    writer.close()
    t = time.time() - start_time
    logger.info(f"All time: {t}")
    logger.info("Finish\n")
