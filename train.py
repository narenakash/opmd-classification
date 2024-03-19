import torch 

import os
import wandb
from tqdm import tqdm

from monai.transforms import Activations, AsDiscrete

from test import test
from utils import save_checkpoint, load_checkpoint, calculate_metrics


def train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, device, save_dir, n_epochs=10, save_freq=1, n_gpus=4, quantize=False):

    best_val_acc = -1
    best_val_acc_epoch = -1

    for epoch in range(n_epochs):

        train_loss, train_precision, train_recall, train_sensitivity, train_specificity, train_f1, train_accuracy = train_epoch(model, train_dataloader, criterion, optimizer, device)

        val_loss, val_precision, val_recall, val_sensitivity, val_specificity, val_f1, val_accuracy = eval_epoch(model, val_dataloader, criterion, device)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_val_acc_epoch = epoch + 1   

            save_checkpoint(model, optimizer, save_dir, epoch + 1)
            print("Saved checkpoint at epoch {} to {}".format(epoch + 1, save_dir))

        print("\nEpoch: {} | Train loss: {:.4f} | Train precision: {:.4f} | Train recall: {:.4f} | Train sensitivity: {:.4f} | Train specificity: {:.4f} | Train f1: {:.4f} | Train acc: {:.4f}\nVal loss: {:.4f} | Val precision: {:.4f} | Val recall: {:.4f} | Val sensitivity: {:.4f} | Val specificity: {:.4f} | Val f1: {:.4f} | Val acc: {:.4f}".format(epoch + 1, train_loss, train_precision, train_recall, train_sensitivity, train_specificity, train_f1, train_accuracy, val_loss, val_precision, val_recall, val_sensitivity, val_specificity, val_f1, val_accuracy))

        if scheduler is not None:
            scheduler.step(val_loss)

        test_loss,  test_precision, test_recall, test_sensitivity, test_specificity, test_f1, test_accuracy = test(model, test_dataloader, criterion, device)
        print("Test loss: {:.4f} | Test precision: {:.4f} | Test recall: {:.4f} | Test sensitivity: {:.4f} | Test specificity: {:.4f} | Test f1: {:.4f} | Test acc: {:.4f}".format(test_loss, test_precision, test_recall, test_sensitivity, test_specificity, test_f1, test_accuracy))

        wandb.log({
            "train_loss": train_loss,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_sensitivity": train_sensitivity,
            "train_specificity": train_specificity,
            "train_f1": train_f1,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_sensitivity": val_sensitivity,
            "val_specificity": val_specificity,
            "val_f1": val_f1,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_acc": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_sensitivity": test_sensitivity,
            "test_specificity": test_specificity,
            "test_f1": test_f1,
            "test_accuracy": test_accuracy,
        })

    # load the best checkpoint
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, os.path.join(save_dir, "model_{}.pth".format(best_val_acc_epoch)), device)
    print("Loaded checkpoint from epoch {} with best val acc {:.4f}".format(start_epoch, best_val_acc))

    test(model, test_dataloader, criterion, device, quantize=quantize)


def train_epoch(model, dataloader, criterion, optimizer, device):
    
    model.train()

    epoch_loss = 0

    labels_list = []
    outputs_list = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        outputs = Activations(sigmoid=True)(outputs)
        outputs = AsDiscrete(threshold=0.5)(outputs)
        outputs = outputs.squeeze()

        labels_list.extend(labels.tolist())
        outputs_list.extend(outputs.tolist())

    epoch_loss /= len(dataloader)

    precision, recall, sensitivity, specificity, f1, accuracy = calculate_metrics(labels_list, outputs_list)

    return epoch_loss, precision, recall, sensitivity, specificity, f1, accuracy


def eval_epoch(model, dataloader, criterion, device):
    
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

            epoch_loss += loss.item()

            outputs = Activations(sigmoid=True)(outputs)
            outputs = AsDiscrete(threshold=0.5)(outputs)
            outputs = outputs.squeeze()

            labels_list.extend(labels.tolist())
            outputs_list.extend(outputs.tolist())

    epoch_loss /= len(dataloader)

    precision, recall, sensitivity, specificity, f1, accuracy = calculate_metrics(labels_list, outputs_list)


    return epoch_loss, precision, recall, sensitivity, specificity, f1, accuracy