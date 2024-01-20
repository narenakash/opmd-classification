import torch

import time
import wandb
from tqdm import tqdm

from monai.transforms import Activations, AsDiscrete, Compose


def test(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0
    epoch_acc = []

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

            epoch_acc.append(acc.item())

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader.dataset)