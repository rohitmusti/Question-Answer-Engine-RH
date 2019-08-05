import numpy as np
import random
import ujson as json
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from toolkit import qcd, collate_fn, get_logger, EMA, get_save_dir, AverageMeter, CheckpointSaver
from args import get_exp3_train_args
from models import classifier

def main(args):

    exp_name = "exp3_train"
    
    # setting up logging
    log = get_logger(args.logging_dir, exp_name)


    # setting a save directory
    save_dir = get_save_dir("./checkpoints", exp_name, training=True, id_max=200)

    # setting up tensor board
    tbx = SummaryWriter(save_dir)
    
    # setting up saver
    saver = CheckpointSaver(save_dir=save_dir, max_checkpoints=args.max_checkpoints,
                            metric_name="BCELoss", log=log)

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
    train_dataset = qcd(data_path=args.train_feature_file, num_categories=args.num_categories)
    train_loader = data.DataLoader(train_dataset,
                                   shuffle=True, 
                                   batch_size=args.batch_size,
                                   collate_fn=collate_fn)
    dev_dataset = qcd(data_path=args.dev_feature_file, num_categories=args.num_categories)
    dev_loader = data.DataLoader(dev_dataset,
                                   shuffle=False, 
                                   batch_size=args.batch_size,
                                   collate_fn=collate_fn)

    # setting up the model
    model = classifier(args=args, word_vectors=word_vectors)
    model = nn.DataParallel(model, gpu_ids)
    model.to(device)
    model.train() 
    ema = EMA(model, args.ema_decay)

    optimizer = optim.Adadelta(model.parameters(), args.learning_rate,
                               weight_decay=args.learning_rate_decay)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR
    step = 0
    steps_till_eval = args.eval_steps

    for epoch in range(100):
        log.info(f"Starting epoch {epoch+1}")
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for qw_idxs, ids, topic_ids, lengths in train_loader:
                qw_idxs = qw_idxs.to(device)
                batch_size = qw_idxs.size(0)
                if batch_size != args.batch_size:
                    log.info('Did not process because did not meet batch_size threshold')
                    continue
                topic_ids = topic_ids.to(device)
                lengths = lengths.to(device)
                optimizer.zero_grad()

                targets = [torch.zeros(args.num_categories) for _ in topic_ids]
                targets = torch.stack(targets)
                for tid, t in zip(topic_ids, targets):
                    t[tid] = 1
                res = model(qw_idxs, lengths)

                # for loss, either nn.softmax_cross_entropy_with_logits or nn.BCELoss or nn.BCEWithLogitsLoss
                # not really sure why this is working and the others aren't
        #        loss = nn.CrossEntropyLoss()
                loss = nn.BCELoss()
       #         loss = nn.BCEWithLogitsLoss()
                loss_output = loss(res, targets)
                loss_output.backward()
                loss_val = loss_output.item()
                optimizer.step()
                scheduler.step(step//batch_size)
                ema(model, step//batch_size)

                step += batch_size
                steps_till_eval -= batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(BCELoss=(loss_val),
                                         Epoch=(epoch + 1))

                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    log.info(f"Evaluating at step: {step}")

                    ema.assign(model)
                    perc_correct, vis_examples, avg_loss = evaluate(model, dev_loader, device,
                                                                    args.dev_eval_file)
                    log.info(f"Out of Sample BCE loss: {avg_loss} at step {step} in epoch {epoch+1}, resulting in {perc_correct} percent correct")

                    tbx.add_scalar("BCE Loss", loss_val, step)
                    tbx.add_scalar("Percent Accuracy", perc_correct, step)


                    for i, example in enumerate(vis_examples):
                        tbl_fmt = (f'- **Question:** {example["question"]}\n'
                                   + f'- **Topic ID:** {example["answer"]}\n'
                                   + f'- **Prediction:** {example["prediction"]}')

                        tbx.add_text(tag=f'{i}_of_{len(vis_examples)}',
                                     text_string=tbl_fmt,
                                     global_step=step)
    
                    saver.save(model=model, step=step, epoch=epoch, 
                               metric_val=loss_val, device=device)

                    # TODO: Finish writing the evaluation script and the tensorboard logging

def evaluate(model, data_loader, device, eval_file):
    pred_dict = {}
    model.eval()
    loss_val_acc = 0
    counter = 0

    with open(eval_file, "r") as fh:
        truth = json.load(fh)
        with torch.no_grad():
            for qw_idxs, ids, topic_ids, lengths in data_loader:

                qw_idxs = qw_idxs.to(device)
                batch_size = qw_idxs.size(0)
                topic_ids = topic_ids.to(device)
                lengths = lengths.to(device)

                targets = [torch.zeros(args.num_categories) for _ in topic_ids]
                targets = torch.stack(targets)
                for tid, t in zip(topic_ids, targets):
                    t[tid] = 1

                res = model(qw_idxs, lengths)

                loss = nn.BCELoss()

                loss_output = loss(res, targets)
                loss_val = loss_output.item()
                loss_val_acc += loss_val
                counter += 1

                temp_pred_dict = {str(int(idx)): int(torch.argmax(i)) for i, idx in zip(res, ids)}

                pred_dict.update(temp_pred_dict)

        correct = 0
        total_covered = 0
        vis_examples = []
        num_cor = 0
        num_incor = 0


        for i in pred_dict:
            if i in truth:
                if truth[i] == pred_dict[i]:
                    if num_cor < 3:
                        vis_examples.append({'question': truth[i]['question'],
                                            'answer': truth[i]['topic_id'],
                                            'prediction':pred_dict[i]})
                        num_cor += 1

                    correct += 1
                else:

                    if num_incor < 3:
                        vis_examples.append({'question': truth[i]['question'],
                                            'answer': truth[i]['topic_id'],
                                            'prediction':pred_dict[i]})
                        num_incor += 1

                total_covered += 1

        perc_correct = correct/total_covered
        average_loss = loss_val_acc/counter

    return perc_correct, vis_examples, average_loss


                





if __name__ == "__main__":
    args = get_exp3_train_args()
    main(args)
    

