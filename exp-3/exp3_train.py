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

from toolkit import qcd, collate_fn
from args import get_exp3_train_args
from models import classifier

def main(args):

    # grabbing a gpu if it is available
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # setting up the model
    with open(args.word_emb_file, 'r') as fh:
        word_vectors = np.array(json.load(fh))
    word_vectors = torch.from_numpy(word_vectors)
    train_dataset = qcd(data_path=args.train_feature_file)
    train_loader = data.DataLoader(train_dataset,
                                   shuffle=True,
                                   batch_size=5,
                                   collate_fn=collate_fn)

    model = classifier(args=args, word_vectors=word_vectors)
    model = nn.DataParallel(model, gpu_ids)
    model.to(device)
    model.train() 

    count = 0 # to be deleted

    # model = model.double()
    
    for qw_idxs, ids, topic_ids in train_loader:
        # qw_idxs = qw_idxs.double()
        qw_idxs = qw_idxs.to(device)
        res = model(qw_idxs)

        print(res.size()) #  to be deleted
        if count > 2: #  to be deleted
            break #  to be deleted
        count += 1 #  to be deleted

if __name__ == "__main__":
    args = get_exp3_train_args()
    main(args)
    

