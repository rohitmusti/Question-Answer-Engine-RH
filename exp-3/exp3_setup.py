import ujson as json
from tqdm import tqdm
from collections import Counter
import spacy
import numpy as np
import re

from toolkit import get_logger, save
from args import get_exp3_featurize_args

def word_tokenize(sent):
    doc = nlp(sent)
    return [re.sub('[^a-zA-Z]', '', token.text.lower().strip()) for token in doc if not token.is_stop]

def pre_process(args, in_file,  word_counter, logger):
    examples = []
    eval_examples = {}
    topic_title_id_map = {}
    total = 0
    logger.info(f"Pre-processing {in_file}")
    with open(in_file, "r") as fh:
        source = json.load(fh)
        for question, topic_id, topic_title in tqdm(source['data']):
            total += 1
            qw_tokens = word_tokenize(question)
            for token in qw_tokens:
                word_counter[token] += 1
            topic_title_id_map[topic_id] = topic_title
            example = {"qw_tokens": qw_tokens,
                       "id": total}
            eval_examples[str(total)] = {"question": question,
                                        "topic_id": topic_id}
            examples.append(example)

    return examples, eval_examples, topic_title_id_map

def get_word_embedding(args, counter, limit=-1, vec_size=300, num_vectors=2196017, logger=None):
    logger.info(f"Pre-processing {args.data_split} vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    with open(args.glove_file, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, total=num_vectors):
            array = line.split()
            word = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            if word in counter and counter[word] > limit:
                embedding_dict[word] = vector
    logger.info(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding embedding vector")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

def featurize(args, examples, out_file, word2idx_dict, data_type, logger=None):
    total, total_ = 0, 0
    ques_idxs, ids = [], []
    print(f"Featurizing {data_type} examples")
    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1
    for example in tqdm(examples):
        total += 1
        ques_idx = np.zeros([args.ques_limit], dtype=np.int32)

        if len(example['qw_tokens']) > args.ques_limit:
            continue

        total_ += 1

        for i, token in enumerate(example["qw_tokens"]):
                ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)
        ids.append(example['id'])

    np.savez(out_file,
             qw_idxs=np.array(ques_idxs),
             ids=np.array(ids))
    logger.info(f"Built and saved {total_}/{total} fully featurized examples")
            
            
if __name__=="__main__":
    global nlp
    args = get_exp3_featurize_args()
    log = get_logger(log_dir=args.logging_dir, name="data-gen")
    nlp = spacy.blank("en")
    word_counter = Counter()
    examples, eval_examples, topic_title_id_map = pre_process(args=args, 
                                                              in_file=args.train_in_file,
                                                              word_counter=word_counter, 
                                                              logger=log)
    save(filename=args.train_topic_title_id_map_file, obj=topic_title_id_map)
    save(filename=args.train_eval_file, obj=eval_examples)

    word_emb_mat, word2idx_dict = get_word_embedding(args=args, counter=word_counter,
                                                     logger=log)
    save(args.word_emb_file, word_emb_mat)
    
    dev_examples, dev_eval_examples, dev_topic_title_id_map = pre_process(args=args, 
                                                              in_file=args.dev_in_file,
                                                              word_counter=word_counter, 
                                                              logger=log)

    save(filename=args.dev_topic_title_id_map_file, obj=dev_topic_title_id_map)
    save(filename=args.dev_eval_file, obj=dev_eval_examples)


    
    featurize(args=args, examples=examples, out_file=args.train_feature_file,
              word2idx_dict=word2idx_dict, data_type="train", logger=log)
    featurize(args=args, examples=dev_examples, out_file=args.dev_feature_file,
              word2idx_dict=word2idx_dict, data_type="dev", logger=log)

