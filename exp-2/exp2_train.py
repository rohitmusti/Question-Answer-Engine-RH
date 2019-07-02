"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from toolkit import get_logger
import config
import sys

from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
import ujson 
from util import collate_fn, SQuAD


def main(c, flags):

    if flags[1] == "train" or flags[1] == "dev":
        word_emb_file = c.word_emb_file
    if flags[1] == "toy":
        word_emb_file = c.toy_word_emb_file
        train_record_file = c.toy_record_file_exp2
        eval_file = c.toy_dev_eval_file
    elif flags[1] == "train":
        train_record_file = c.train_record_file_exp2
        eval_file = c.dev_eval_file
    elif flags[1] == "dev":
        train_record_file = c.dev_record_file_exp2
        eval_file = c.toy_eval_file
    else:
        raise ValueError("Unregonized or missing flag")

    # Set up logging and devices
    name = "train exp2"
    c.save_dir = util.get_save_dir(c.logging_dir, name, training=True)
    log = get_logger(c.save_dir, name)
    tbx = SummaryWriter(c.save_dir)
    device, c.gpu_ids = util.get_available_devices()
    log.info(f"Args: {dumps(vars(c), indent=4, sort_keys=True)}")
    c.batch_size *= max(1, len(c.gpu_ids))

    # Set random seed
    log.info(f"Using random seed {c.random_seed}...")
    random.seed(c.random_seed)
    np.random.seed(c.random_seed)
    torch.manual_seed(c.random_seed)
    torch.cuda.manual_seed_all(c.random_seed)

    # Get embeddings
    log.info("Loading embeddings...")
    word_vectors = util.torch_from_json(word_emb_file)

    # Get model
    log.info("Building model...")
    model = BiDAF(word_vectors=word_vectors,
                  hidden_size=c.hidden_size,
                  drop_prob=c.drop_prob)
    model = nn.DataParallel(model, c.gpu_ids)
    if c.load_path:
        log.info(f"Loading checkpoint from {c.load_path}...")
        model, step = util.load_model(model, c.load_path, c.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, c.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(c.save_dir,
                                 max_checkpoints=c.max_checkpoints,
                                 metric_name=c.metric_name,
                                 maximize_metric=c.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), c.learning_rate,
                               weight_decay=c.learning_weight_decay)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(train_record_file, use_v2=True)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=c.batch_size,
                                   shuffle=True,
                                   num_workers=c.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = SQuAD(c.dev_record_file_exp2, use_v2=True)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=c.batch_size,
                                 shuffle=False,
                                 num_workers=c.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = c.eval_steps
    epoch = step // len(train_dataset)
    # revert to: while epoch != c.num_epochs:
    torch.set_num_threads(7)
    while epoch != 2:
        epoch += 1
        log.info(f"Starting epoch {epoch}...")
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                log_p1, log_p2 = model(cw_idxs, qw_idxs)
                y1, y2 = y1.to(device), y2.to(device)
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                # revert to: if steps_till_eval <= 0:
                if True:
                    steps_till_eval = c.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f"Evaluating at step {step}...")
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  eval_file,
                                                  c.max_ans_len,
                                                  use_squad_v2=True)
                    saver.save(step, model, results[c.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f"Dev {results_str}")

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f"dev/{k}", v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=c.num_visuals)
                log.info("finished one epoch")


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = ujson.load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model(cw_idxs, qw_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    c = config.config()
    flags = sys.argv
    main(c, flags)
