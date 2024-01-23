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

        # num_neurons = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(num_neurons, num_classes)

        # from the source code of the MDPI paper
        self.model.classifier = nn.Sequential(
            # nn.AvgPool2d((1,1)),
            # nn.Flatten(),
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

# print the model
if __name__ == '__main__':
    model = DenseNet201()
    print(model)
    print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))