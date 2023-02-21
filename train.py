import argparse

import torch
import torchvision.transforms as transforms
from torchmetrics import Accuracy

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from model import TestResNet, PlainCNN
from dataset import Cifar10DataLoader
from utils import fix_seed, get_now

fix_seed()

def train():
    device = f"cuda:{args.device_id}" if args.cuda else "cpu"
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # dataset
    train_dataloader, valid_dataloader = Cifar10DataLoader(train=True, transform=transform).get_dataloader(args.batch_size, args.num_workers)
    
    # metric
    metric_acc = Accuracy(task='multiclass', num_classes=10).to(device)
    
    # model
    if args.model == 'plain_cnn':
        model = PlainCNN().to(device)
    elif args.model == 'resnet':
        model = TestResNet(init_layers=args.init_layers).to(device)
    else:
        raise Exception("Unsupported model")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min', factor=0.5, verbose=True)
    
    if args.mlflow:
        run_name = args.run_name if args.run_name is not None else f"test_{get_now(True)}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            'init_lr': args.lr,
            'model': args.model
        })
    
    for epoch in range(args.epochs):
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        
        model.train()
        
        step = 0
        running_loss = 0
        running_acc = 0
        
        for batch_idx, (img_batch, target_batch) in enumerate(train_dataloader):
            
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
            
            step += 1
            
            status = (
                "\r> epoch: {:3d} > step: {:3d} > loss: {:.3f}, lr: {:.3f}, acc: {:.2f}".format(
                    epoch+1, step, running_loss/(batch_idx+1), current_lr, running_acc/(batch_idx+1)
                )
            )
            
            print(status, end="")
        train_loss = running_loss / len(train_dataloader)
        train_acc = running_acc / len(train_dataloader)
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
        
        scheduler.step(metrics=val_loss)
        
        if args.mlflow:
            mlflow.log_metrics({
                'train_accuracy': train_acc.numpy().item(),
                'train_loss': train_loss.numpy().item(),
                'validation_accuracy': val_acc.numpy().item(),
                'validation_loss': val_loss.numpy().item(),
                'lr': current_lr
            })
    
    if args.mlflow:
        # input_schema = Schema([
        #     TensorSpec(np.dtype(np.float32), (-1, 3, 32, 32))
        # ])
        # output_schema = Schema([
        #     TensorSpec(np.dtype(np.float32), (-1, 10))
        # ])
        
        conda_env = {
            'name': 'mlflow-env',
            'channels': ['conda-forge'],
            'dependencies': [
                'python=3.9.16',
                'pip<=22.3.1',
                {'pip': [
                    'mlflow<3,>=2.1',
                    'cloudpickle==2.2.1',
                    'ipython==8.10.0',
                    'torch==1.12.1',
                    'torchvision==0.13.1',
                    '--extra-index-url https://download.pytorch.org/whl/cu113',
                    'torchmetrics==0.11.1',
                ]}
            ],
        }
        
        mlflow.pytorch.log_model(
            model,
            artifact_path="test_resnet",
            # signature=ModelSignature(inputs=input_schema, outputs=output_schema),
            conda_env=conda_env
        )
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
        mlflow.end_run()

def test():
    
    pass

def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--init_layers", type=int, default=1)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--model", type=str, default='plain_cnn')
    parser.add_argument("--run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    train()
    