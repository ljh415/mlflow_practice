import os
import time
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import mlflow
from torchmetrics import Accuracy

from model import TestResNet
from utils import fix_seed

fix_seed()

def train():
    
    # print(vars(params))
    # sys.exit()
    
    device = f"cuda:{args.device_id}" if args.cuda else "cpu"
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # dataset
    data = datasets.CIFAR10(root='~/data', train=True, transform=transform, download=True)
    train_data, valid_data = random_split(data, [int(len(data)*0.8), len(data)-int(len(data)*0.8)])
    
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    metric_acc = Accuracy(task='multiclass', num_classes=10).to(device)
    
    # model
    model = TestResNet(init_layers=args.init_layers).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, mode='min', factor=0.1, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.001)
    
    for epoch in range(args.epochs):
        model.train()
        
        step = 0
        running_loss = 0
        running_acc = 0
        
        for batch_idx, (img_batch, target_batch) in enumerate(train_dataloader):
            # now_lr = args.lr
            now_lr = scheduler.optimizer.param_groups[0]['lr']
            
            optimizer.zero_grad()
            
            img = img_batch.to(device)
            target = target_batch.to(device)
            
            pred = model(img)

            loss = criterion(pred, target)
            acc = metric_acc(pred, target)
            
            running_loss += loss.detach().cpu()
            running_acc += acc.detach().cpu()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            step += 1
            
            status = (
                "\r> epoch: {:3d} > step: {:3d} > loss: {:.3f}, lr: {:.3f}, acc: {:.2f}".format(
                    epoch+1, step, running_loss/(batch_idx+1), now_lr, running_acc/(batch_idx+1)
                )
            )
            
            print(status, end="")
            
        print()
        ## validate
        
        running_val_loss = 0
        running_val_acc = 0
        
        with torch.no_grad():
            model.eval()
            for batch_idx, (img_batch, target_batch) in enumerate(valid_dataloader):
                
                img = img_batch.to(device)
                target = target_batch.to(device)
                
                pred = model(img)
                
                val_loss = criterion(pred, target)
                val_acc = metric_acc(pred, target)
                
                running_val_loss += val_loss.detach().cpu()
                running_val_acc += val_acc.detach().cpu()
                
                status = (
                    "\r validation : {:6d} / {:6d}".format(
                        batch_idx+1,
                        len(valid_dataloader)
                    )
                )
                print(status, end="")
        
        val_loss = running_val_loss / len(valid_dataloader)
        val_acc = running_val_acc / len(valid_dataloader)
        print("\nValidation loss: {:3f}, acc: {:.2f}\n".format(val_loss, val_acc))
        
    
    # print(model)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--init_layers", type=int, default=1)
    
    args = parser.parse_args()    
    # params = Params(args.batch_size, args.epochs, args.lr, args.num_workers, args.device_id, args.cuda)
    
    train()
    
    pass