import os
import torch
import argparse

class Params(object):
    def __init__(self, batch_size:int, epochs:int, lr:float, num_workers:int, device_id:int=None, cuda:bool=False):
        assert num_workers < os.cpu_count() 
        assert device_id is not None and cuda
        if cuda:
            assert torch.cuda.is_available()
            
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.cuda = cuda
        self.num_workers = num_workers
        self.device_id = device_id