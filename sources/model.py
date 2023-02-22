from collections import OrderedDict
from itertools import islice

import torch
import torch.nn as nn
from torchvision.models import resnet50
from sources.dataset import Cifar10DataLoader

class PlainCNN(nn.Module):
    """
    Just plain Convolution Network using cifar10 dataset
    """
    def __init__(self) -> None:
        super(PlainCNN, self).__init__()
        # model
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(3200, 10)
        )
        
    def forward(self, x:torch.Tensor):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x

class TestResNet(nn.Module):
    """
    Using ResNet50, cifar10 dataset only
    """
    def __init__(self, init_layers:int=1, num_classes:int=10) -> None:
        super(TestResNet, self).__init__()
        
        self.resnet50_backbone = resnet50()
        self.resnet50_backbone = nn.Sequential(OrderedDict(islice(self.resnet50_backbone._modules.items(), len(self.resnet50_backbone._modules)-2)))
        
        self.init_layers = init_layers
        self.avg_pool =  nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
        
        self._initialize()
        
    def _initialize(self) -> None:
        """
        reset parameter
        """
        for idx, (_, module) in enumerate(self.resnet50_backbone._modules.items()):
            if idx < len(self.resnet50_backbone) - self.init_layers:
                continue
            module.apply(self._init_weight)
                
    def _init_weight(self, layer) -> None:
        """
        reset parameter
        """
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    
    def forward(self, x):
        out = self.resnet50_backbone(x)
        out = self.avg_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        out = torch.softmax(out, dim=0)
        
        return out