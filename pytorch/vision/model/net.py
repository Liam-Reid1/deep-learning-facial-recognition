"""Modified ResNet-18 (proposal): standard blocks + ReLU, global average pooling, FC to one output per person."""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if getattr(params, 'pretrained', True) else None
        self.resnet = models.resnet18(weights=weights)  # ~11.2M parameters
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, params.num_classes)

    def forward(self, images):
        # [batch, 3, height, width] -> [batch, num_classes] raw logits.
        return self.resnet(images)


def loss_fn(outputs, labels):
    # CrossEntropyLoss expects raw logits, not softmax probabilities.
    return F.cross_entropy(outputs, labels)


def accuracy(outputs, labels):
    return float(np.mean(np.argmax(outputs, axis=1) == labels))


metrics = {'accuracy': accuracy}
