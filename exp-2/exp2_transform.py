import ujson as json
from toolkit import get_logger, quick_clean, save
from config import config
import sys
from random import randrange
from tqdm import tqdm

def exp2_transformer(in_file, out_file, logger):
    new_data = {}
    new_data["experiment"] = 1
    counter = 0
    with open(in_file, "r") as fh:
        logger.info(f"Importing {fh.name}")
        source = json.load(fh)
        new_data["version"] = source["version"]
        new_data["data"] = []
        logger.info("Creating all context list")
        for topic_id, topic in tqdm(enumerate(source["data"])):
            logger.info(f"Processing: {topic['title']}")
            context_buffer = 0
            topic_dict = {}
            topic_dict["title"] = topic["title"]
            topic_dict["qas"] = []
            topic_contexts = [para["context"] for para in topic["paragraphs"]]
            topic_dict["context"] = " ".join(topic_contexts)
            for para in topic["paragraphs"]:
                paragraph = {}
                paragraph["qas"] = []
                for qas in para['qas']:
                    counter += 1
                    qas_dict = {}
                    qas_dict["topic_id"] = topic_id
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
                    topic_dict["qas"].append(qas_dict)
                context_buffer += len(para['context']) + 1
            new_data["data"].append(topic_dict)

    logger.info(f"Processed {counter} question, answer pairs")
    logger.info(f"Saving to {out_file}")
    save(filename=out_file, obj=new_data)

if __name__ == "__main__":
    flags = sys.argv
    c = config()
    logger = get_logger(log_dir=c.logging_dir, name="exp2 data transformer")
    valid_args = ["test", "train", "dev", "toy", "all"]

    if flags[1] not in valid_args:
        logger.info("Not a valid args")
        logger.info(f"Valid args are: {valid_args}")
    else:

        if flags[1]=="test":
            c, b = get_new_context("test", ["test1", "test2", "test3", "test4", "test5",
                            "test6", "test7", "test8", "test9"])
            logger.info(f"New context: {c}")
            logger.info(f"Index of start of 'test': {b}")
            test = "test" == c[b:b+4]
            logger.info(f"Checking if the indexes line up: {test}")
        if flags[1]=="train" or flags[1]=="all":
            exp2_transformer(c.train_data_orig, c.train_data_exp2, logger)
        if flags[1]=="dev" or flags[1]=="all":
            exp2_transformer(c.dev_data_orig, c.dev_data_exp2, logger)
        if flags[1]=="toy" or flags[1]=="all":
            exp2_transformer(c.toy_data_orig, c.toy_data_exp2, logger)
