import os
import numpy as np 
import pandas as pd 

from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


# dataset for OPMD classification with images of two classes stored in csv
class OPMDClassificationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.csv_file.iloc[idx, 0])
        
        image = Image.open(img_name).convert('RGB')
        image = np.asarray(image, dtype=np.float32)
        
        label = self.csv_file.iloc[idx, 1]
        label = np.asarray(label, dtype=np.float32)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return {"image": image, "label": label} 
    

# unit-test the dataset
if __name__ == '__main__':
    root_dir = '/ssd_scratch/cvit/chocolite/OPMD-GraceSet/images'
    csv_file = './data/opmd_train_fold1.csv'
    
    # test the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = OPMDClassificationDataset(csv_file, root_dir, transform=None)
    print(len(dataset))

    print(dataset[0]["image"].shape)
    print(dataset[0]["label"])