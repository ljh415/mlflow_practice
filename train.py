import os
import time
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, random_split

import mlflow.pytorch
from torchmetrics import Accuracy

def train():
    
    transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    
    # dataset
    data = datasets.CIFAR100(root='~/data', train=True, transform=transform)
    train_data, valid_data = random_split(data, [len(data)*0.8, len(data)*0.2])
    test_data = datasets.CIFAR100(root='~/data', train=False)
    
    train_dataloder = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    valid_dataloder = DataLoader(valid_data, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    
    metric = Accuracy()
    
    # model
    
    # train
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    pass