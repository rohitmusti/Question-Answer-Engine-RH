"""
tldr: modifies the squad data to fit the requirements of my experiment 2 (topic contexts)

author: @rohitmusti
"""
import ujson as json
from tqdm import tqdm
from toolkit import save, quick_clean, get_logger
from random import randrange
from args import get_data_gen_args

def toy_transformer(raw_data_file, train_data_file, dev_data_file, test_data_file, train_topic_num, dev_topic_num, test_topic_num, logger):
    """
    distill original data into at most 15 topics, with each having at most 5 paragraphs,
    each of which has 5 questions and 5 answers
    args:
        - raw_data_file: the file name of the data to be transformed to experiment 2
        - out_file: the file name of where the ought to be written

    return:
        none, the data is written to an output
    """
    logger.info(f"This toy data set will be compromised of {train_topic_num + test_topic_num + dev_topic_num} topics")
    new_train_data = {}
    new_train_data['experiment'] = "toy"
    new_dev_data = {}
    new_dev_data['experiment'] = "toy_dev"
    new_test_data = {}
    new_test_data['experiment'] = "toy_train"
    with open(raw_data_file, "r") as fh:
        logger.info(f"Importing: {raw_data_file}")
        source = json.load(fh)
        logger.info("Converting into toy format")
        new_train_data["version"] = source["version"]
        new_train_data["data"] = []
        new_dev_data["version"] = source["version"]
        new_dev_data["data"] = []
        new_test_data["version"] = source["version"]
        new_test_data["data"] = []
        train_topic_counter = train_topic_num
        dev_topic_counter = dev_topic_num
        test_topic_counter = test_topic_num
        for topic in (source["data"]):
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

            if train_topic_counter > 0:
                new_train_data["data"].append(topic_dict)
            elif dev_topic_counter > 0:
                new_dev_data["data"].append(topic_dict)
            elif test_topic_counter > 0:
                new_test_data["data"].append(topic_dict)
            else:
                break

            if train_topic_counter >= 0:
                train_topic_counter -= 1
            elif dev_topic_counter >= 0:
                dev_topic_counter -= 1
            elif test_topic_counter >= 0:
                test_topic_counter -= 1
            else:
                break

    logger.info(f"Saving new data to {train_data_file}")
    save(filename=train_data_file, obj=new_train_data)
    logger.info(f"Saving new dev data to {dev_data_file}")
    save(filename=dev_data_file, obj=new_dev_data)
    logger.info(f"Saving new test data to {test_data_file}")
    save(filename=test_data_file, obj=new_test_data)

if __name__ == "__main__":
    args = get_data_gen_args()
    log = get_logger(log_dir=args.logging_dir, name="data-gen")
    toy_transformer(raw_data_file=args.raw_data,
                    train_data_file=args.train_data_orig,
                    dev_data_file=args.dev_data_orig,
                    test_data_file=args.test_data_orig,
                    train_topic_num=args.train_topic_num, 
                    dev_topic_num=args.dev_topic_num,
                    test_topic_num=args.test_topic_num, 
                    logger=log)
