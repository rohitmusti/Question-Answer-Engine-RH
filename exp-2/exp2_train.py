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
import sys

from args import get_exp2_training_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
import ujson 
from util import collate_fn, SQuAD


def main(args):

    # Set up logging and devices
    name = "train_exp2"
    args.save_dir = util.get_save_dir(args.logging_dir, name, training=True)
    log = get_logger(args.save_dir, name)
    tbx = SummaryWriter(args.save_dir)
    device, gpu_ids = util.get_available_devices()
    log.info(f"Args: {dumps(vars(args), indent=4, sort_keys=True)}")
    args.batch_size *= max(1, len(gpu_ids))

    # Set random seed
    log.info(f"Using random seed {args.random_seed}...")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # Get embeddings
    log.info(f"Loading embeddings from {args.word_emb_file}...")
    word_vectors = util.torch_from_json(args.word_emb_file)

    # Get model
    log.info("Building model...")
    model = BiDAF(word_vectors=word_vectors,
                  hidden_size=args.hidden_size,
                  drop_prob=args.drop_prob)
    model = nn.DataParallel(model, gpu_ids)
    if args.load_path:
        log.info(f"Loading checkpoint from {args.load_path}...")
        model, step = util.load_model(model, args.load_path, gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.learning_rate,
                               weight_decay=args.learning_rate_decay)
    # scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR
    scheduler = sched.ReduceLROnPlateau(optimizer=optimizer,
                                        mode="min", factor=0.1,
                                        patience=2, verbose=True, cooldown=0 
                                        min_lr=0.0005)


    for epoch in range(args.num_epochs):
        log.info(f"Starting epoch {epoch}...")
        for i in range(args.num_train_chunks):
        # Get data loader
            train_rec_file = f"{args.train_record_file_exp2}_{i}.npz"
            log.info(f'Building dataset from {train_rec_file} ...')
            train_dataset = SQuAD(train_rec_file, args.exp2_train_topic_contexts, use_v2=True)
            train_loader = data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           collate_fn=collate_fn)

            # Train
            log.info('Training...')
            steps_till_eval = args.eval_steps
            epoch = 0
        # torch.set_num_threads(7)
            with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
                for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                    # Setup for forward
                    cw_idxs = cw_idxs.to(device)
                    qw_idxs = qw_idxs.to(device)
                    batch_size = qw_idxs.size(0)
                    optimizer.zero_grad()

                    # Forward
                    log_p1, log_p2 = model(cw_idxs, qw_idxs)
                    y1, y2 = y1.to(device), y2.to(device)
                    loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                    loss_val = loss.item()

                    # Backward
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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
                    if steps_till_eval <= 0:
                        steps_till_eval = args.eval_steps

                        # Evaluate and save checkpoint
                        log.info(f"Evaluating at step {step}...")
                        ema.assign(model)

                        for i in range(args.num_dev_chunks):
                        # Get data loader
                            all_pred_dicts = {}
                            all_results = OrderedDict() 
                            dev_rec_file = f"{args.dev_record_file_exp2}_{i}.npz"
                            log.info(f'Building evaluating dataset from {dev_rec_file} ...')
                            dev_dataset = SQuAD(dev_rec_file, 
                                                args.exp2_dev_topic_contexts, 
                                                use_v2=True)
                            dev_loader = data.DataLoader(dev_dataset,
                                                           batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers,
                                                           collate_fn=collate_fn)
                            results, pred_dict = evaluate(model, dev_loader, device,
                                                          args.dev_eval_file,
                                                          args.max_ans_len,
                                                          use_squad_v2=True)
                            all_results.update(results)
                            all_pred_dicts.update(pred_dict)

                            del dev_dataset
                            del dev_loader
                            del results
                            del pred_dict
                            torch.cuda.empty_cache()

                        saver.save(step, model, all_results[args.metric_name], device)
                        ema.resume(model)

                        # Log to console
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in all_results.items())
                        log.info(f"Dev {results_str}")

                        # Log to TensorBoard
                        log.info('Visualizing in TensorBoard...')
                        for k, v in all_results.items():
                            tbx.add_scalar(f"dev/{k}", v, step)
                        util.visualize(tbx,
                                       pred_dict=all_pred_dicts,
                                       eval_path=args.dev_eval_file,
                                       step=step,
                                       split='dev',
                                       num_visuals=args.num_visuals)
                    torch.cuda.empty_cache()
            del train_dataset
            del train_loader
            torch.cuda.empty_cache()

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
    args = get_exp2_training_args()
    main(args)
