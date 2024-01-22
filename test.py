import torch
import numpy as np

import time
import wandb
from tqdm import tqdm

from monai.transforms import Activations, AsDiscrete
from sklearn.metrics import classification_report, confusion_matrix

def test(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0
    epoch_acc = []

    labels_list = []
    outputs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

            outputs = Activations(sigmoid=True)(outputs)
            outputs = AsDiscrete(threshold_values=True)(outputs)
            outputs = outputs.squeeze()
            acc = (outputs == labels).float().tolist()

            epoch_loss += loss.item()

            epoch_acc.extend(acc)

            labels_list.extend(labels.tolist())
            outputs_list.extend(outputs.tolist())

        print('Classification report:')
        print(classification_report(np.asarray(labels_list), np.asarray(outputs_list)))

        print('Confusion matrix:')
        print(confusion_matrix(np.asarray(labels_list), np.asarray(outputs_list)))

        epoch_loss /= len(dataloader)

        epoch_acc = sum(epoch_acc)
        epoch_acc /= len(dataloader.dataset)

    return epoch_loss, epoch_acc