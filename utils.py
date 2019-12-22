import datetime
import numpy as np
import os
import pickle
import random
import torch

def set_seed(seed):
    '''
    Ensure reproducibility.
    '''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_file(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)

def load_file(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)

def remove_file(fpath):
    try:
        os.remove(fpath)
    except OSError:
        pass

def write(fpath, text):
    with open(fpath, 'a+') as f:
        f.write(text + '\n')

def get_time():
    return datetime.datetime.now().strftime('%H:%M:%S')