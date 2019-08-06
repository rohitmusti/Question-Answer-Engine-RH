"""
tldr: modifies the squad data to fit the requirements of my experiment 2 (topic contexts)

author: @rohitmusti
"""
import ujson as json
from tqdm import tqdm
from toolkit import save, quick_clean, get_logger
from random import randrange
from args import get_exp2_data_gen_args

def toy_transformer(in_file, train_file, dev_file, test_file, train_topic_num, dev_topic_num, test_topic_num, logger):
    """
    distill original data into at most 15 topics, with each having at most 5 paragraphs,
    each of which has 5 questions and 5 answers
    args:
        - in_file: the file name of the data to be transformed to experiment 2
        - out_file: the file name of where the ought to be written

    return:
        none, the data is written to an output
    """
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
        topic_counter = train_topic_num
        for topic in tqdm(source["data"]):
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
            elif topic_counter >= -1*dev_topic_num:
                new_dev_data["data"].append(topic_dict)
            elif topic_counter >= -1*(dev_topic_num+test_topic_num):
                new_test_data["data"].append(topic_dict)
            else:
                break

            topic_counter -= 1

    logger.info(f"Saving new data to {train_file}")
    save(filename=train_file, obj=new_data)
    logger.info(f"Saving new dev data to {dev_file}")
    save(filename=dev_file, obj=new_dev_data)
    logger.info(f"Saving new test data to {test_file}")
    save(filename=test_file, obj=new_test_data)

if __name__ == "__main__":
    args = get_exp2_data_gen_args()
    log = get_logger(log_dir=args.logging_dir, name="data-gen")
    toy_transformer(in_file=args.raw_train_data, 
                    train_file=args.train_data_src, 
                    dev_file=args.dev_data_src,
                    test_file=args.test_data_src,
                    train_topic_num=args.train_topic_num,
                    dev_topic_num=args.dev_topic_num,
                    test_topic_num=args.test_topic_num,
                    logger=log)
