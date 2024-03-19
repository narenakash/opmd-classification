import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from monai.transforms import Activations, AsDiscrete
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm

from models.densenet import DenseNet201
from dataset2 import OPMDClassificationDataset
from utils import get_config, set_seed, calculate_metrics

config = get_config('config.yaml')


set_seed(config['seed'])

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

test_transforms = A.Compose([
            A.Resize(config['resize'], config['resize']),
            A.CenterCrop(config['crop'], config['crop']),
            ToTensorV2(),
        ])

def get_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_buffers = sum(b.numel() for b in model.buffers())
    total_size = num_params * 4 + num_buffers * 4  # Assuming 32-bit floating point
    return total_size

test_dataset = OPMDClassificationDataset('data/opmd_test_fold1.csv', '/ssd_scratch/cvit/chocolite/OPMD-GraceSet/images', transform=test_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = DenseNet201(num_classes=1, quantize=config['quantize']).to(device)
model = nn.DataParallel(model)

model.load_state_dict(torch.load('/ssd_scratch/cvit/chocolite/OPMD-GraceClassification/weights/densenet201_fold_1_dataaug_1_nchannel_3_lr_0.0001_bs_32_epochs_50_19-03-2024_08-33-06/model_44.pth')['model_state_dict'])

model = model.module.to(device)


# RGB, fold1: /ssd_scratch/cvit/chocolite/OPMD-GraceClassification/weights/densenet201_fold_1_dataaug_1_nchannel_3_lr_0.0001_bs_32_epochs_50_13-02-2024_02-56-51/model_44.pth

# RGB, fold1, seg train, seg test: /ssd_scratch/cvit/chocolite/OPMD-GraceClassification/weights/densenet201_fold_1_dataaug_1_nchannel_3_lr_0.0001_bs_32_epochs_50_13-02-2024_10-29-08/model_11.pth

model.eval()

print(get_model_size(model))

if config['quantize']:
    model.qconfig = torch.quantization.get_default_qconfig('x86')
# quantized_model = torch.quantization.quantize_static(model, qconfig_spec={torch.nn.Linear: torch.quantization.default_linear_quant_config})


img_name_list = []
labels_list = []
outputs_list = []

# fuse the densenet201 model inplace for static quantization
# if config['quantize']:
    # modules_to_fuse = [['module.model.features.conv0', 'module.model.features.norm0', 'module.model.features.relu0'],
                    #  ['module.model.features.denseblock1.denselayer1.norm1', 'module.model.features.denseblock1.denselayer1.relu1', 'module.model.features.denseblock1.denselayer1.conv1'],
#                      ['module.model.features.denseblock1.denselayer1.norm2', 'module.model.features.denseblock1.denselayer1.relu2', 'module.model.features.denseblock1.denselayer1.conv2'],
#                      ['module.model.features.denseblock1.denselayer2.norm1', 'module.model.features.denseblock1.denselayer2.relu1', 'module.model.features.denseblock1.denselayer2.conv1'],
#                      ['module.model.features.denseblock1.denselayer2.norm2', 'module.model.features.denseblock1.denselayer2.relu2', 'module.model.features.denseblock1.denselayer2.conv2'],
#                      ['module.model.features.denseblock1.denselayer3.norm1', 'module.model.features.denseblock1.denselayer3.relu1', 'module.model.features.denseblock1.denselayer3.conv1'],
#                      ['module.model.features.denseblock1.denselayer3.norm2', 'module.model.features.denseblock1.denselayer3.relu2', 'module.model.features.denseblock1.denselayer3.conv2'],
#                      ['module.model.features.denseblock1.denselayer4.norm1', 'module.model.features.denseblock1.denselayer4.relu1', 'module.model.features.denseblock1.denselayer4.conv1'],
#                      ['module.model.features.denseblock1.denselayer4.norm2', 'module.model.features.denseblock1.denselayer4.relu2', 'module.model.features.denseblock1.denselayer4.conv2'],
#                      ['module.model.features.denseblock1.denselayer5.norm1', 'module.model.features.denseblock1.denselayer5.relu1', 'module.model.features.denseblock1.denselayer5.conv1'],
#                      ['module.model.features.denseblock1.denselayer5.norm2', 'module.model.features.denseblock1.denselayer5.relu2', 'module.model.features.denseblock1.denselayer5.conv2'],
#                      ['module.model.features.denseblock1.denselayer6.norm1', 'module.model.features.denseblock1.denselayer6.relu1', 'module.model.features.denseblock1.denselayer6.conv1'],
#                      ['module.model.features.denseblock1.denselayer6.norm2', 'module.model.features.denseblock1.denselayer6.relu2', 'module.model.features.denseblock1.denselayer6.conv2'],                     
                #    ]
    # model = torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
    

# prepare the model for static quantization
if config['quantize']:
    model = torch.quantization.prepare(model)
    torch.quantization.convert(model, inplace=True)
    print(get_model_size(model))

# test the model

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

    torch.save(model.state_dict(), "temp.p")
    print('Size of the model(MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

# store each image's name, label and output
# with open('output/fold1_results.csv', 'w') as f:
#     f.write('image_name,label,output\n')
#     for i in range(len(img_name_list)):
#         f.write(f'{img_name_list[i]},{labels_list[i]},{outputs_list[i]}\n')