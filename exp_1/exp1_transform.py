import ujson as json
from toolkit import get_logger, quick_clean, save
from config import config
import sys
from random import randrange
from tqdm import tqdm

def get_new_context(orig_context, all_contexts):
    """
    Given an original context, return it , randomly placed,
    with 10 other randomly picked contexts, along with the index it starts at

    Args:
        orig_context (str): the original contexts to be embedded in the other contexts
        all_contexts (str): all of the contexts

    Returns:
        new_context: the new context formed from the 10 random contexts + the orig one
        context_buffer: the buffer amount of all the other contexts so that the answers line up
    """

    indexes = [randrange(0, len(all_contexts)) for i in range(10)]
    insert_index = randrange(0, 10)

    context_set = [str(all_contexts[i] + " ") for i in indexes]
    context_set_lens = [len(i) for i in context_set]
    context_buffer = sum(context_set_lens[:insert_index])
    context_set.insert(insert_index, str(orig_context+" "))
    new_context = "".join(context_set)

    return new_context, context_buffer

def exp_1_transformer(in_file, out_file, logger):
    new_data = {}
    new_data["experiment"] = 1
    counter = 0
    with open(in_file, "r") as fh:
        logger.info("Importing {}".format(fh.name))
        source = json.load(fh)
        new_data["version"] = source["version"]
        new_data["data"] = []
        logger.info("Creating all context list")
        all_contexts = [para["context"] for topic in source["data"] for para in topic["paragraphs"]]
        for topic in tqdm(source["data"]):
            logger.info("Processing: {}".format(topic["title"]))
            topic_dict = {}
            topic_dict["title"] = topic["title"]
            topic_dict["paragraphs"] = []
            for para in topic["paragraphs"]:
                paragraph = {}
                paragraph["context"], context_buffer = get_new_context(orig_context=para["context"], all_contexts=all_contexts)
                paragraph["qas"] = []
                for qas in para['qas']:
                    counter += 1
                    qas_dict = {}
                    qas_dict["id"] = qas["id"]
                    qas_dict["is_impossible"] = qas["is_impossible"]
                    qas_dict["question"] = quick_clean(raw_str=qas["question"])
                    qas_dict["answers"] = []
                    if not qas["is_impossible"]:
                        for answer in qas["answers"]:
                            answer_dict = {}
                            answer_dict["answer_start"] = context_buffer + answer["answer_start"]
                            answer_dict["text"] = answer["text"]

                            qas_dict["answers"].append(answer_dict)
                    paragraph["qas"].append(qas_dict)
                topic_dict["paragraphs"].append(paragraph)
            new_data["data"].append(topic_dict)

    logger.info("Processed {} question, answer pairs".format(counter))
    logger.info("Saving to {}".format(out_file))
    save(filename=out_file, obj=new_data)

if __name__ == "__main__":
    flags = sys.argv
    c = config()
    logger = get_logger(log_dir=c.logging_dir, name="exp_1 data transformer")
    valid_args = ["test", "train", "dev", "toy", "all"]

    if flags[1] not in valid_args:
        print("Not a valid args")
        print("Valid args are: {}".format(valid_args))
    else:

        if flags[1]=="test":
            c, b = get_new_context("test", ["test1", "test2", "test3", "test4", "test5",
                            "test6", "test7", "test8", "test9"])
            print("New context: {}".format(c))
            print("Index of start of 'test': {}".format(b))
            test = "test" == c[b:b+4]
            print("Checking if the indexes line up: {}".format(test))
        if flags[1]=="train" or flags[1]=="all":
            exp_1_transformer(c.train_data_orig, c.train_data_exp1, logger)
        if flags[1]=="dev" or flags[1]=="all":
            exp_1_transformer(c.dev_data_orig, c.dev_data_exp1, logger)
        if flags[1]=="toy" or flags[1]=="all":
            exp_1_transformer(c.toy_data_orig, c.toy_data_exp1, logger)
