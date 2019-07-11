"""
tldr: modifies the squad data to fit the requirements of my experiment 2 (topic contexts)

author: @rohitmusti
"""
import ujson as json
from tqdm import tqdm
from toolkit import save, quick_clean, get_logger
import sys
from random import randrange
import config

def toy_transformer(in_file, out_file_1, out_file_2, out_file_3, topic_num, logger):
    """
    distill original data into at most 15 topics, with each having at most 5 paragraphs,
    each of which has 5 questions and 5 answers
    args:
        - in_file: the file name of the data to be transformed to experiment 2
        - out_file: the file name of where the ought to be written

    return:
        none, the data is written to an output
    """
    logger.info(f"This toy data set will be compromised of {topic_num} topics")
    new_data = {}
    new_data['experiment'] = "toy"
    new_dev_data = {}
    new_dev_data['experiment'] = "toy_dev"
    new_test_data = {}
    new_test_data['experiment'] = "toy_train"
    with open(in_file, "r") as fh:
        logger.info(f"Importing: {in_file}")
        source = json.load(fh)
        logger.info("Converting into toy format")
        new_data["version"] = source["version"]
        new_data["data"] = []
        new_dev_data["version"] = source["version"]
        new_dev_data["data"] = []
        new_test_data["version"] = source["version"]
        new_test_data["data"] = []
        topic_counter = topic_num
        for topic in tqdm(source["data"]):
            logger.info(f"Processing: {topic['title']}")
            topic_dict = {}
            topic_dict["title"] = topic["title"]
            topic_dict["paragraphs"] = []
            for para in topic["paragraphs"]:
                paragraph = {}
                paragraph["context"] = para["context"]
                paragraph["qas"] = []
                for qas in para['qas']:
                    qas_dict = {}
                    qas_dict["id"] = qas["id"]
                    qas_dict["is_impossible"] = qas["is_impossible"]
                    qas_dict["question"] = quick_clean(raw_str=qas["question"])
                    qas_dict["answers"] = []
                    if not qas["is_impossible"]:
                        for answer in qas["answers"]:
                            answer_dict = {}
                            answer_dict["answer_start"] = answer["answer_start"]
                            answer_dict["text"] = answer["text"]
                            qas_dict["answers"].append(answer_dict)
                    paragraph["qas"].append(qas_dict)
                topic_dict["paragraphs"].append(paragraph)
            if topic_counter >= 0:
                new_data["data"].append(topic_dict)
            elif topic_counter >= -20:
                new_dev_data["data"].append(topic_dict)
            else:
                new_test_data["data"].append(topic_dict)
            if topic_counter == -30:
                break
            topic_counter -= 1

    logger.info(f"Saving new data to {out_file_1}")
    save(filename=out_file_1, obj=new_data)
    logger.info(f"Saving new dev data to {out_file_2}")
    save(filename=out_file_2, obj=new_dev_data)
    logger.info(f"Saving new test data to {out_file_3}")
    save(filename=out_file_3, obj=new_test_data)

if __name__ == "__main__":
    c = config.config()
    flags = sys.argv
    topic_num = int(flags[1])
    log = get_logger(log_dir=c.logging_dir, name="data-gen")
    toy_transformer(in_file="data/train/orig-train-v2.0.json", 
                    out_file_1="data/train/train-v2.0.json", 
                    out_file_2="data/dev/dev-v2.0.json", 
                    out_file_3="data/test/test-v2.0.json", 
                    topic_num=topic_num,logger=log)
