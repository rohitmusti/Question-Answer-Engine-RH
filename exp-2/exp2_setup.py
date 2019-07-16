"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import os
import sys
import spacy
import ujson as json
import urllib.request

from args import get_exp2_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile
from toolkit import save, get_logger, quick_clean

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            logger.info(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter, logger):
    logger.info(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    topic_context_examples = []
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for topic_id, topic in tqdm(enumerate(source["data"])):

            # context processing
            topic_context = quick_clean(raw_str=topic["context"])
            topic_context_tokens = word_tokenize(topic_context)
            topic_context_chars = [list(token) for token in topic_context_tokens]
            spans = convert_idx(topic_context, topic_context_tokens)
            for token in topic_context_tokens:
                # it was originally len(para['qas']) but that seemed arbitrary so I used 1s
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1

            topic_context_dict = {"context_tokens":topic_context_tokens,
                                  "context_chars":topic_context_chars,
                                  "context":topic_context}
            topic_context_examples.append(topic_context_dict)

            # qas processing
            for qa in topic["qas"]:
                total += 1

                #question processing
                ques = quick_clean(qa["question"])
                ques_tokens = word_tokenize(ques)
                ques_chars = [list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1

                # answer processing
                y1s, y2s = [], []
                answer_texts = []
                for answer in qa["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                example = {"ques_tokens": ques_tokens,
                           "ques_chars": ques_chars,
                           "topic_context_id": topic_id,
                           "y1s": y1s,
                           "y2s": y2s,
                           "id": total}
                examples.append(example)
                eval_examples[str(total)] = {"question": ques,
                                             "context": topic_context_examples[topic_id]["context"],
                                             "spans": spans,
                                             "answers": answer_texts,
                                             "uuid": qa["id"]}

        logger.info(f"{len(examples)} questions in total")
    return examples, eval_examples, topic_context_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    logger.info(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        logger.info(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding {data_type} embedding vector")
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        logger.info(f"{len(filtered_elements)} tokens have corresponding {data_type} embedding vector")

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

def is_answerable(example):
    return len(example['y2s']) > 0 and len(example['y1s']) > 0

def build_features(c, examples, topic_contexts, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    topic_context_lens = [len(context["context"]) for context in  topic_contexts]
    para_limit = max(topic_context_lens)
    ques_limit = c.ques_limit
    ans_limit = c.ans_limit
    char_limit = c.char_limit
    del topic_context_lens

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(topic_contexts[ex["topic_context_id"]]["context_tokens"]) > para_limit or \
                   len(ex["ques_tokens"]) > ques_limit or \
                   (is_answerable(ex) and
                    ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    topic_ids = []
    y1s = []
    y2s = []
    ids = []

    # context feature building
    logger.info(f"Creating the {data_type} topic_context features")
    for topic in tqdm(topic_contexts):
        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(topic["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(topic["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

    # question + answer feature building
    logger.info(f"Creating the {data_type} question and answer features")
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)


        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])
        topic_ids.append(example["topic_context_id"])


    np.savez(out_file,
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids),
             topic_ids=np.array(topic_ids))
    logger.info(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta


def pre_process(args, logger):
    # Process training set and use it to decide on the word/character vocabularies

    word_counter, char_counter = Counter(), Counter()
    examples, eval_obj, topic_contexts_examples = process_file(filename=args.train_exp2_data, 
                                                               data_type="train", 
                                                               word_counter=word_counter, 
                                                               char_counter=char_counter, 
                                                               logger=logger)

    save(args.train_eval_file, eval_obj)
    del eval_obj

    word_emb_mat, word2idx_dict = get_embedding(word_counter, 'word', 
                                                emb_file=args.glove_word_file, 
                                                vec_size=args.glove_word_dim, 
                                                num_vectors=args.glove_word_num_vecs)
    char_emb_mat, char2idx_dict = get_embedding(char_counter, 'char', 
                                                emb_file=None, 
                                                vec_size=args.glove_char_dim)

    save(args.word_emb_file, word_emb_mat)
    save(args.char_emb_file, char_emb_mat)
    del word_emb_mat
    del char_emb_mat

    save(args.word2idx_file, word2idx_dict)
    save(args.char2idx_file, char2idx_dict)

    build_features(c=args, examples=examples,topic_contexts=topic_contexts_examples,
                   data_type="train", out_file=args.train_record_file_exp2, 
                   word2idx_dict=word2idx_dict, char2idx_dict=char2idx_dict, is_test=False)
    del topic_contexts_examples
    del examples

    # Process dev and test sets
    dev_examples, dev_eval, dev_topic_contexts = process_file(filename=args.dev_data_exp2, 
                                                              data_type="dev", 
                                                              word_counter=word_counter, 
                                                              char_counter=char_counter, 
                                                              logger=logger)
    dev_meta = build_features(c=args, examples=dev_examples, 
                              topic_contexts=dev_topic_contexts, 
                              data_type="dev", 
                              out_file=args.dev_record_file_exp2, 
                              word2idx_dict=word2idx_dict, 
                              char2idx_dict=char2idx_dict)
    del dev_topic_contexts
    del dev_examples

    save(args.dev_eval_file, dev_eval)
    save(args.dev_meta_file, dev_meta)

if __name__ == '__main__':
    # Get command-line args
    args = get_exp2_setup_args()

    # Import spacy language model
    nlp = spacy.blank("en")

    # get flags
    flags = sys.argv

    # set up logger
    logger = get_logger(log_dir=args.logging_dir, name="exp2 setup")

    # Preprocess dataset
    pre_process(args=args, logger=logger)
