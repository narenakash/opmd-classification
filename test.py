import torch
import numpy as np

from tqdm import tqdm

from monai.transforms import Activations, AsDiscrete
from sklearn.metrics import classification_report, confusion_matrix

from utils import calculate_metrics


def test(model, dataloader, criterion, device, quantize=False):
    model.eval()

    epoch_loss = 0

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
            outputs = AsDiscrete(threshold=0.5)(outputs)
            outputs = outputs.squeeze()

            epoch_loss += loss.item()

            labels_list.extend(labels.tolist())
            outputs_list.extend(outputs.tolist())

        print('Classification report:')
        print(classification_report(np.asarray(labels_list), np.asarray(outputs_list)))

        print('Confusion matrix:')
        print(confusion_matrix(np.asarray(labels_list), np.asarray(outputs_list)))

        epoch_loss /= len(dataloader)

        precision, recall, sensitivity, specificity, f1, accuracy = calculate_metrics(labels_list, outputs_list)

        print('Test metrics:')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}\n')

    return epoch_loss, precision, recall, sensitivity, specificity, f1, accuracy