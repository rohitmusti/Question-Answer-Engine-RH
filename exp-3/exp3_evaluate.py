import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data

from models import classifier
from args import get_exp3_train_args
from toolkit import get_logger, qcd, collate_fn
import ujson as json

def main(args):
    # setting up logging
    logger = get_logger(log_dir=args.logging_dir, name="exp3_evaluation")

    # grabbing GPU
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    logger.info(f"Using device type: {device}")

    # getting word embeddings
    with open(args.word_emb_file, 'r') as fh:
        word_vectors = np.array(json.load(fh))
    word_vectors = torch.from_numpy(word_vectors)

    # loading in the model
    model = classifier(args=args, word_vectors=word_vectors)
    model = nn.DataParallel(model, gpu_ids)

    ckpt_dict = torch.load("./checkpoints/train/exp3_train-34/best.pth.tar", 
                           map_location=device)

    model.load_state_dict(ckpt_dict['model_state'])


    dataset = qcd(data_path=args.dev_feature_file, num_categories=args.num_categories)
    loader = data.DataLoader(dataset,
                             shuffle=True, 
                             batch_size=args.batch_size,
                             collate_fn=collate_fn)

    # loading eval_file
    with open(args.dev_eval_file, 'r') as fh:
        gold_dict = json.load(fh)
        all_predicted_indexes = {}
        predicted_indexes = {}
        with torch.no_grad():
            for qw_idxs, ids, topic_ids, lengths in loader:

                qw_idxs.to(device)
                ids.to(device)
                topic_ids.to(device)

                batch_size = qw_idxs.size(0)

                if batch_size != args.batch_size:
                    logger.info('Did not process because did not meet batch_size threshold')
                    continue

                targets = [torch.zeros(args.num_categories) for _ in topic_ids]
                targets = torch.stack(targets)
                for tid, t in zip(topic_ids, targets):
                    t[tid] = 1

                res = model(qw_idxs, lengths)

                predicted_indexes = {int(idx): int(torch.argmax(i)) for i, idx in zip(res, ids)}
                all_predicted_indexes.update(predicted_indexes)

        print(f"Was able to predict {len(all_predicted_indexes)}/{len(gold_dict)} total examples.")

        correct = 0
        total_eval = 0
        for i in all_predicted_indexes:
            if i in gold_dict:
                if all_predicted_indexes[i] == gold_dict[i]:
                    correct += 1
                total_eval += 1
        logger.info(f"Got {correct}/{total_eval} correct")




if __name__ == "__main__":
    args = get_exp3_train_args()
    main(args)