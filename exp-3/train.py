import numpy as np
import random
import ujson as json

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from args import get_exp3_train_args

def main(args):
    
    with open(args.word_emb_file, 'r') as fh:
        word_vectors = np.array(json.load(fh))
    word_vectors = torch.from_numpy(word_vectors).type(np.int32)
