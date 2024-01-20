import torch 

import os
import wandb
from tqdm import tqdm

from monai.transforms import Activations, AsDiscrete

from test import test
from utils import save_checkpoint, load_checkpoint


def train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, device, save_dir, n_epochs=config['epochs'], save_freq=1, n_gpus=4):

    best_val_acc = -1
    best_val_acc_epoch = -1

    for epoch in range(n_epochs):

        train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, scheduler, device)

        val_loss, val_acc = eval_epoch(model, val_dataloader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch + 1   

            save_checkpoint(model, optimizer, save_dir, epoch + 1)
            print("Saved checkpoint at epoch {} to {}".format(epoch + 1, save_dir))

        print("Current epoch: {} | Train loss: {:.4f} | Train acc: {:.4f} | Val loss: {:.4f} | Val acc: {:.4f} | Best val acc: {:.4f} at epoch {}".format(epoch + 1, train_loss, train_acc, val_loss, val_acc, best_val_acc, best_val_acc_epoch))

        test_loss, test_acc = test(model, test_dataloader, criterion, device)
        print("Test loss: {:.4f} | Test acc: {:.4f}".format(test_loss, test_acc))

        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "test_loss": test_loss, "test_acc": test_acc, "epoch": epoch + 1, "best_val_acc": best_val_acc, "best_val_acc_epoch": best_val_acc_epoch})

    # load the best checkpoint
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, os.path.join(save_dir, "model_{}.pth".format(best_val_acc_epoch)), device)
    print("Loaded checkpoint from epoch {} with best val acc {:.4f}".format(start_epoch, best_val_acc))

    test(model, test_dataloader, criterion, device)


