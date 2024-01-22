import torch
import torch.nn as nn

import torchvision.models as models
from torchvision.models import DenseNet201_Weights



class DenseNet201(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet201, self).__init__()

        self.model = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        num_neurons = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_neurons, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x