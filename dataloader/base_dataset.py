import os
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image
import csv

class BaseDataset(data.Dataset):
    def __init__(self, cfg, phase="train"):
        super().__init__()
	raise NotImplementedError
	pass

    def __len__(self):
	raise NotImplementedError
	pass

    def __getitem__(self, index):
	raise NotImplementedError
	pass
