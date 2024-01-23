import os
import wandb
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.densenet import DenseNet201
from dataset import OPMDClassificationDataset
from utils import get_config, set_seed

from train import train

config = get_config('config.yaml')



def main(run_name):

    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config['data_aug'] == 1:
        print('Data augmentation is enabled')
        train_transforms = A.Compose([
            A.Resize(config['resize'], config['resize']),
            A.CenterCrop(config['crop'], config['crop']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(p=0.25),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ])
    else:
        print('Data augmentation is disabled')
        train_transforms = A.Compose([
            A.Resize(config['resize'], config['resize']),
            A.CenterCrop(config['crop'], config['crop']),
            A.Normalize(),
            ToTensorV2(),
        ])
        
    val_transforms = A.Compose([
            A.Resize(config['resize'], config['resize']),
            A.CenterCrop(config['crop'], config['crop']),
            ToTensorV2(),
        ])
    test_transforms = A.Compose([
            A.Resize(config['resize'], config['resize']),
            A.CenterCrop(config['crop'], config['crop']),
            ToTensorV2(),
        ])

    train_dataset = OPMDClassificationDataset(config['train_dir'], transform=train_transforms)
    val_dataset = OPMDClassificationDataset(config['val_dir'], transform=val_transforms)
    test_dataset = OPMDClassificationDataset(config['test_dir'], transform=test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    model = DenseNet201(num_classes=config['num_classes']).to(device)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    scheduler = None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-8, verbose=True)


    save_dir = os.path.join(config['save_dir'], run_name)

    train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, device, save_dir, n_epochs=config['n_epochs'], save_freq=1, n_gpus=4)



if __name__ == '__main__':

    project_name = config['project_name']

    run_name = f"{config['model']['name']}_fold_{config['fold']}_dataaug_{config['data_aug']}_nchannel_{config['model']['n_channels']}_lr_{config['init_lr']}_bs_{config['batch_size']}_epochs_{config['n_epochs']}"
    run_name = run_name + "_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    wandb.init(
        project=project_name,
        config=config,
        entity="chocolite",
        name=run_name,
        mode="disabled"    
    )

    wandb.config.update(config)

    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    save_dir = os.path.join(config["save_dir"], run_name)
    os.makedirs(save_dir)

    main(run_name=run_name)

    wandb.finish()