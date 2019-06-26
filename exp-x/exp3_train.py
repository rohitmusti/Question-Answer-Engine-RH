"""Train a model on SQuAD.

Author:
    Rohit Musti (rmusti@redhat.com)
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as torchdata
import util
import config
import sys

from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
import ujson as json
from util import collate_fn, SQuAD3


def main(data, flags):
    # Set up logging and devices
    log_dir = data.logging_dir
    log = util.get_logger(log_dir, "toy")
    tbx = SummaryWriter(data.logging_dir)
    device, data.gpu_ids = util.get_available_devices()
    log.info('Config: {}'.format(dumps(vars(data), indent=4, sort_keys=True)))
    data.batch_size *= max(1, len(data.gpu_ids))

    # Set random seed
    log.info('Using random seed {}...'.format(data.random_seed))
    random.seed(data.random_seed)
    np.random.seed(data.random_seed)
    torch.manual_seed(data.random_seed)
    torch.cuda.manual_seed_all(data.random_seed)

    if flags[1] == "toy":
        word_emb_file = data.toy_word_emb_file
        training_data = data.toy_record_file_exp3
        test_data = data.dev_record_file_exp3
        eval_file = data.toy_eval_exp3
    elif flags[1] == "train":
        word_emb_file = data.word_emb_file
        training_data = data.train_record_file_exp3
        test_data = data.dev_record_file_exp3
        eval_file = data.train_eval_exp3
    elif flags[1] == "dev":
        word_emb_file = data.word_emb_file
        training_data = data.dev_record_file_exp3
        test_data = data.toy_record_file_exp3
        eval_file = data.dev_eval_exp3

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(word_emb_file)

    # Get model
    log.info('Building model...')
    model = BiDAF(word_vectors=word_vectors,
                  hidden_size=data.hidden_size,
                  drop_prob=data.drop_prob)
    model = nn.DataParallel(model, data.gpu_ids)
    if data.load_path:
        log.info('Loading checkpoint from {}...'.format(data.load_path))
        model, step = util.load_model(model, data.load_path, data.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, data.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(data.logging_dir,
                                 max_checkpoints=10,
                                 metric_name=data.metric_name,
                                 maximize_metric=data.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), data.learning_rate,
                               weight_decay=data.learning_weight_decay)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    # np.load(data.toy_record_file_exp3)
    train_dataset = SQuAD3(training_data, use_v2=True)
    train_loader = torchdata.DataLoader(train_dataset,
                                   batch_size=data.batch_size,
                                   shuffle=True,
                                   num_workers=data.num_workers,
                                   collate_fn=collate_fn)

    test_dataset = SQuAD3(test_data, use_v2=True)
    test_loader = torchdata.DataLoader(test_dataset,
                                 batch_size=data.batch_size,
                                 shuffle=False,
                                 num_workers=data.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = data.eval_steps
    epoch = step // len(test_dataset)
    while epoch != data.num_epochs:
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                log.info("cw_idxs length: {}".format(str(len(cw_idxs))))
                log.info("qw_idxs length: {}".format(str(len(qw_idxs))))
                log.info("cw_idxs size: {}".format(str(sys.getsizeof(cw_idxs))))
                log.info("qw_idxs size: {}".format(str(sys.getsizeof(qw_idxs))))
                log.info("cw_idxs shape: {}".format(str(cw_idxs.shape)))
                log.info("qw_idxs shape: {}".format(str(qw_idxs.shape)))

                log_p1, log_p2 = model(cw_idxs, qw_idxs)
                y1, y2 = y1.to(device), y2.to(device)
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), data.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('toy/NLL', loss_val, step)
                tbx.add_scalar('toy/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = data.eval_steps

                    # Evaluate and save checkpoint
                    log.info('Evaluating at step {}...'.format(step))
                    ema.assign(model)
                    results, pred_dict = evaluate(model, test_loader, device,
                                                  eval_path=eval_file,
                                                  max_len=sys.maxsize,
                                                  use_squad_v2=True)
                    saver.save(step, model, results[data.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                            for k, v in results.items())
                    log.info('Dev {}'.format(results_str))

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar('dev/{}'.format(k), v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=data.num_visuals)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json.load(fh)
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
    main(data=config.data(), flags=sys.argv)
