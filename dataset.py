import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import Dataset

class Cifar100Dataset(Dataset):
    def __init__(self, data_path:str = '~/base', transform:torchvision.transforms.Compose = None) -> None:
        super(Cifar100Dataset).__init__()
        self.cifar100_data = datasets.CIFAR100(root=data_path, train=True)
        self.cifar100_test_data = datasets.CIFAR100(root=data_path, train=False)
    
    def __len__(self):
        return len(self)
    
    pass

cifar100_data = datasets.CIFAR100(root='~/data', download=True)
