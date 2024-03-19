import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from monai.transforms import Activations, AsDiscrete
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm

from models.densenet import DenseNet201
from dataset2 import OPMDClassificationDataset
from utils import get_config, set_seed, calculate_metrics

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

config = get_config('config.yaml')


set_seed(config['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_transforms = A.Compose([
            A.Resize(config['resize'], config['resize']),
            A.CenterCrop(config['crop'], config['crop']),
            ToTensorV2(),
        ])

test_dataset = OPMDClassificationDataset('data/opmd_test_fold1.csv', '/ssd_scratch/cvit/chocolite/OPMD-GraceSegClassification-F1-N3/images', transform=test_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = DenseNet201(num_classes=1).to(device)
model = nn.DataParallel(model)

model.load_state_dict(torch.load('/ssd_scratch/cvit/chocolite/OPMD-GraceClassification/weights/densenet201_fold_1_dataaug_1_nchannel_3_lr_0.0001_bs_32_epochs_50_13-02-2024_10-29-08/model_11.pth')['model_state_dict'])

# RGB, fold1: /ssd_scratch/cvit/chocolite/OPMD-GraceClassification/weights/densenet201_fold_1_dataaug_1_nchannel_3_lr_0.0001_bs_32_epochs_50_13-02-2024_02-56-51/model_44.pth

# RGB, fold1, seg train, seg test: /ssd_scratch/cvit/chocolite/OPMD-GraceClassification/weights/densenet201_fold_1_dataaug_1_nchannel_3_lr_0.0001_bs_32_epochs_50_13-02-2024_10-29-08/model_11.pth

target_layers = model.module.model.features[-1]
cam = GradCAM(model=model, target_layers=target_layers)

model.eval()

img_name_list = []
labels_list = []
outputs_list = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        img_name = batch['image_name']
        img_name_list.append(img_name)

        outputs = model(images)
        outputs = outputs.squeeze()

        outputs = Activations(sigmoid=True)(outputs)
        outputs = AsDiscrete(threshold=0.5)(outputs)
        outputs = outputs.squeeze()

        labels_list.append(labels.tolist())
        outputs_list.append(outputs.tolist())


    print('Classification report:')
    print(classification_report(np.asarray(labels_list), np.asarray(outputs_list)))

    print('Confusion matrix:')
    print(confusion_matrix(np.asarray(labels_list), np.asarray(outputs_list)))

    precision, recall, sensitivity, specificity, f1, accuracy = calculate_metrics(labels_list, outputs_list)

    print('Test metrics:')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}\n')

# store each image's name, label and output
# with open('output/fold1_results.csv', 'w') as f:
#     f.write('image_name,label,output\n')
#     for i in range(len(img_name_list)):
#         f.write(f'{img_name_list[i]},{labels_list[i]},{outputs_list[i]}\n')