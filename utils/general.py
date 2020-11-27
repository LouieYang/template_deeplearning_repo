import torch
import numpy as np
import random

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_gpu(gpu):
    gpu_list = [int(x) for x in gpu.split(',')]
    print ('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    return gpu_list.__len__()
