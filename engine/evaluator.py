import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.general import set_seed

class Evaluator(object):
    def __init__(self, cfg):

        self.cfg = cfg

    def run(self):
        set_seed(self.cfg.seed)

