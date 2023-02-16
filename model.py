from collections import defaultdict, OrderedDict
from itertools import islice

import torch
import torch.nn as nn
from torchvision.models import resnet50

class TestResNet(nn.Module):
    """
    Using ResNet50, cifar100 dataset only

    Args:
        nn (_type_): _description_
    """
    def __init__(self, init_layers:int = 1, num_classes:int = 10) -> None:
        super(TestResNet, self).__init__()
        
        self.resnet50_backbone = resnet50()
        self.resnet50_backbone = nn.Sequential(OrderedDict(islice(self.resnet50_backbone._modules.items(), len(self.resnet50_backbone._modules)-2)))
        
        self.init_layers = init_layers
        self.avg_pool =  nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
        # self.classifier = nn.Sequential(OrderedDict({
        #         ('avgpool', nn.AdaptiveAvgPool2d((1,1))),
        #         ('fc', nn.Linear(2048, num_classes))
        #     }))
        
        self._initialize()
        
    
    def _initialize(self) -> None:
        for idx, (_, module) in enumerate(self.resnet50_backbone._modules.items()):
            if idx < len(self.resnet50_backbone) - self.init_layers:
                continue
            module.apply(self._init_weight)
    
    def _init_weight(self, layer) -> None:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    
    def forward(self, x):
        out = self.resnet50_backbone(x)
        out = self.avg_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        out = torch.softmax(out, dim=0)
        
        return out