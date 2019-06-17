"""
Pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Rohit Musti (rmusti@redhat.com)
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import os
import spacy
import ujson as json
import urllib.request
import config
from toolkit import quick_clean

from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile


def word_tokenize(sent):
    """
    unchanged from @chrischute
    """
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    """
    unchanged from @chrischute
    """
    current = 0
    spans = []
    for token in tokens:
        for small_token in token:
        # unclear why this is necessary, but I'll just appease it for now.
        # the functions should all be sending and receiving the same type
        # but for some reason word_tokenize is receiving a string and sending
        # back a list of lists.
            current = text.find(small_token, current)
            if current < 0:
                print("Token {} cannot be found".format(small_token))
                raise Exception()
            spans.append((current, current + len(small_token)))
            current += len(small_token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    """
    modified to fit the super context experiment
    author: @rohitmusti
    """
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        sc = source["super_context"]
        sc_tokens = word_tokenize(sc)
        sc_chars = [list(token) for token in sc_tokens]
        sc_spans = convert_idx(sc, sc_tokens)
        print("Creating the word indices from the contexts")
        print()
        for token in tqdm(sc_tokens):
            word_counter[token] += 1
            # I changed this to 1 instead of the len(para["qas"]) it was originally.
            for char in token:
                char_counter[char] += 1
            # I changed this to 1 instead of the len(para["qas"]) it was originally.
        examples.append({"super_context_tokens": sc_tokens,
                         "super_context_chars": sc_chars,
                         "spans": sc_spans})
        print()
        print("Pre-processing {} examples...".format(data_type))
        for topic in tqdm(source["data"]):
            for qas in topic["qas"]:
                total += 1
                ques = quick_clean(qas["question"])
                ques_tokens = word_tokenize(ques)
                ques_chars = [list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1
                y1s, y2s = [], []
                answer_texts = []
                for answer in qas["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer["answer_start"]
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []
                    if not qas["is_impossible"]:
                        [answer_span.append(idx) for idx, span in enumerate(sc_spans)]
#                        for idx, span in enumerate(sc_spans):
#                            if not (answer_end <= span[0] or answer_start >= span[1]):
# I may come to regret this comment but I think it is useful for now
# hopefully the is_impossible check helps
#                            answer_span.append(idx)

                        y1, y2 = answer_span[0], answer_span[-1]
                    else:
                        y1, y2 = -1, -1 #signifying no answer
                    y1s.append(y1)
                    y2s.append(y2)

                example = {"ques_tokens": ques_tokens,
                           "ques_chars": ques_chars,
                           "y1s": y1s,
                           "y2s": y2s,
                           "id": total}
                examples.append(example)
                eval_examples[str(total)] = {"question": ques,
                                             "answers": answer_texts,
                                             "uuid": qas["id"]}
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples
#        #####
#        # add context merging here
#        # I need some way to keep track of each article and use that to make the buffer, should be as easy as adding a counter`
#        #####
#
#        #####
#        # once this is done, the above code can probably be re-used
#        #####
#
#                for qa in para["qas"]:
#                    total += 1
#                    ques = qa["question"].replace(
#                    "''", '" ').replace("``", '" ')
#                    ques_tokens = word_tokenize(ques)
#                    ques_chars = [list(token) for token in ques_tokens]
#                    for token in ques_tokens:
#                        word_counter[token] += 1
#                    for char in token:
#                        char_counter[char] += 1
#                    y1s, y2s = [], []
#                    answer_texts = []
#                    for answer in qa["answers"]:
#                        answer_text = answer["text"]
#                        answer_start = answer['answer_start']
#                        answer_end = answer_start + len(answer_text)
#                        answer_texts.append(answer_text)
#                        answer_span = []
#                        for idx, span in enumerate(spans):
#                            if not (answer_end <= span[0] or answer_start >= span[1]):
#                                answer_span.append(idx)
#                        y1, y2 = answer_span[0], answer_span[-1]
#                        # adding the buffer allows us to keep track within the super context
#                        y1s.append(y1 + buffer)
#                        y2s.append(y2 + buffer)
#
#        #####
#        # can probably modify the two below data structures to take
#        # advantage of the smart new context stuff
#        #####
#
#                example = {
#                       # "context_tokens": context_tokens, # removing this to avoid repeats
#                       # "context_chars": context_chars, # removing this to avoid repeats
#                       "ques_tokens": ques_tokens,
#                       "ques_chars": ques_chars,
#                       "y1s": y1s,
#                       "y2s": y2s,
#                       "id": total}
#                examples.append(example)
#                eval_examples[str(total)] = {
#                            # "context": context, # removing this to avoid repeats
#                             "question": ques,
#                             "spans": spans,
#                             "answers": answer_texts,
#                             "uuid": qa["id"]}
#                # we do this at the end so the first buffer doesn't get thrown off
#                buffer += len(context)
#        #####
#        # end mod
#        #####
#
#    print("{} questions in total".format(len(examples)))
#    return examples, eval_examples, super_context


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    """
    TODO: modify to fit the super context experiment
    NOTE: it looks like there actually aren't changes needed to be made here for the super context idea. I will need to examine the function calls to be sure.
    """
    print("Pre-processing {} vectors...".format(data_type))
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
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
            scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding {} embedding vector".format(
            len(filtered_elements), data_type))

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


def convert_to_features(args, data, word2idx_dict, char2idx_dict, is_test):
    """
    TODO: modify to fit the super context experiment
    """
    example = {}

    #####
    # Since I am re-using the same context, I can probably hard code some optimization here
    #####

    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    #####
    # same here
    #####
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    char_limit = args.char_limit

    def filter_func(example):
    #####
    # room for mod here
    #####
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    #####
    # room for reuse here
    #####
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
    #####
    # room for reuse here
    #####
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    #####
    # room for reuse here
    #####
    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    #####
    # room for reuse here; definitely need to modify this return statement to make the memory better
    #####
    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def is_answerable(example):
    """
    unchanged from @chrischute
    """
    return len(example['y2s']) > 0 and len(example['y1s']) > 0


def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    """
    TODO: modify to fit the super context experiment
    a little bit less urgent since it is used for metas, not really important yet
    """
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(ex["context_tokens"]) > para_limit or \
               len(ex["ques_tokens"]) > ques_limit or \
               (is_answerable(ex) and
                ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    print("Converting {} examples to indices...".format(data_type))
    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

    context_idx = np.zeros([para_limit], dtype=np.int32)
    context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idx = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

    #####
    # room for reuse here
    #####
    for i, token in enumerate(example["context_tokens"]):
        context_idx[i] = _get_word(token)
    context_idxs.append(context_idx)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idx[i] = _get_word(token)
    ques_idxs.append(ques_idx)

    #####
    # room for reuse here
    #####
    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idx[i, j] = _get_char(char)
    context_char_idxs.append(context_char_idx)

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

    #####
    # room for space optimization
    #####
    np.savez(out_file,
         context_idxs=np.array(context_idxs),
         context_char_idxs=np.array(context_char_idxs),
         ques_idxs=np.array(ques_idxs),
         ques_char_idxs=np.array(ques_char_idxs),
         y1s=np.array(y1s),
         y2s=np.array(y2s),
         ids=np.array(ids))
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(data):
    """
    unchanged from @chrischute
    """

    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval, train_super_context = process_file(data.train_data_exp3, "train", word_counter, char_counter)
#    word_emb_mat, word2idx_dict = get_embedding(word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
#    char_emb_mat, char2idx_dict = get_embedding(char_counter, 'char', emb_file=None, vec_size=args.char_dim)
#
#    # Process dev and test sets
#    dev_examples, dev_eval, dev_super_context = process_file(args.dev_file, "dev", word_counter, char_counter)
#    build_features(args, train_examples, "train", args.train_record_file, word2idx_dict, char2idx_dict)
#    dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)
#    if args.include_test_examples:
#        test_examples, test_eval = process_file(args.test_file, "test", word_counter, char_counter)
#        save(args.test_eval_file, test_eval, message="test eval")
#        test_meta = build_features(args, test_examples, "test",
#                       args.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
#        save(args.test_meta_file, test_meta, message="test meta")
#
#    save(args.word_emb_file, word_emb_mat, message="word embedding")
#    save(args.char_emb_file, char_emb_mat, message="char embedding")
    save(data.train_eval_exp3, train_eval, message="train eval")
#    save(args.dev_eval_file, dev_eval, message="dev eval")
#    save(args.word2idx_file, word2idx_dict, message="word dictionary")
#    save(args.char2idx_file, char2idx_dict, message="char dictionary")
#    save(args.dev_meta_file, dev_meta, message="dev meta")


if __name__ == '__main__':
    # Get command-line args
    data = config.data()

    # Import spacy language model
    nlp = spacy.blank("en")
    nlp.max_length = 100000000

    # Preprocess dataset
    pre_process(data)