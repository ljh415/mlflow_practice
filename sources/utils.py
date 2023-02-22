import random
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn

def fix_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    
def get_now(time=False):
    now = datetime.datetime.now()
    if time:
        return now.strftime("%y%m%d-%H%M")
    return now.strftime("%y%m%d")