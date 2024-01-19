import os
import numpy as np 
import pandas as pd 

from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


# images of the two classes are stored in two different folders

class OPMDClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = os.listdir(self.root_dir)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        self.imgs = []
        for c in self.classes:
            class_path = os.path.join(self.root_dir, c)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                self.imgs.append((img_path, self.class_to_idx[c]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    

# unit-test the dataset
if __name__ == '__main__':
    root_dir = '/ssd_scratch/cvit/chocolite/OPMD-Classification/train'
    
    # test the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = OPMDClassificationDataset(root_dir, transform=transform)
    print(len(dataset))

    print(dataset[0][0].shape)
    print(dataset[0][1])
    print(dataset[1][0].shape)
    print(dataset[1][1])
    print(dataset[2][0].shape)
    print(dataset[2][1])

    # test the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for imgs, labels in dataloader:
        print(imgs.shape)
        print(labels.shape)
        break