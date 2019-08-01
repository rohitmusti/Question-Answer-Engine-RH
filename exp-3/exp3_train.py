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

from toolkit import qcd, collate_fn, get_logger
from args import get_exp3_train_args
from models import classifier

def main(args):
    
    # setting up logging
    log = get_logger(args.logging_dir, "exp3_train")

    # setting the random seed
    log.info(f"Using random seed {args.random_seed}...")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # grabbing a gpu if it is available
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    log.info(f"Using device type: {device}")

    # getting word embeddings
    with open(args.word_emb_file, 'r') as fh:
        word_vectors = np.array(json.load(fh))
    word_vectors = torch.from_numpy(word_vectors)

    # setting up the datasets
    train_dataset = qcd(data_path=args.train_feature_file)
    train_loader = data.DataLoader(train_dataset,
                                   shuffle=True, # should eventually be reset to True
                                   batch_size=args.batch_size,
                                   collate_fn=collate_fn)

    # setting up the model
    model = classifier(args=args, word_vectors=word_vectors)
    model = nn.DataParallel(model, gpu_ids)
    model.to(device)
    model.train() 

    count = 0 # to be deleted

    for qw_idxs, ids, topic_ids, lengths in train_loader:
        # qw_idxs = qw_idxs.double()
        print(qw_idxs.size())
        print(ids.size())
        print(topic_ids.size())
        qw_idxs = qw_idxs.to(device)
        topic_ids = topic_ids.to(device)
        lengths = lengths.to(device)
        targets = [torch.zeros(442) for _ in topic_ids]
        targets = torch.stack(targets)
        for tid, t in zip(topic_ids, targets):
            t[tid] = 1
#        print(targets)
        res = model(qw_idxs, lengths)


        # for loss, either nn.softmax_cross_entropy_with_logits or nn.BCELoss

#        print(res) # to be deleted
        count += 1 # to be deleted
        if count > 2: # to be deleted
            break # to be deleted

if __name__ == "__main__":
    args = get_exp3_train_args()
    main(args)
    

