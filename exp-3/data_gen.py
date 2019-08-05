from args import get_exp3_transformer_args
from toolkit import get_logger, quick_clean, save
import ujson as json
from tqdm import tqdm

def exp3_transformer(in_file_1, in_file_2,  train_out_file, test_out_file, logger):
    """
    convert data into (question, topic_id, topic title) format
    args:
        - in_file: the file name of the data to be transformed to experiment 3 format
        - out_file: the file name of where the ought to be written

    return:
        none, the data is written to an output
    """
    new_data = {}
    new_data["experiment"] = "toy"
    q_count = 0
    with open(in_file_1, "r") as fh:
        logger.info(f"Importing: {in_file}")
        source = json.load(fh)
        train_new_data["version"] = source["version"]
        train_new_data["data"] = []
        test_new_data["version"] = source["version"]
        test_new_data["data"] = []
        for topic_id, topic in tqdm(enumerate(source["data"])):
            test_count = 0
            for para in topic["paragraphs"]:
                for qas in para['qas']:
                    if test_count < 5:
                        test_count += 1
                        test_new_data["data"].append((quick_clean(raw_str=qas["question"]), topic_id, topic["title"]))
                    else:
                        train_data["data"].append((quick_clean(raw_str=qas["question"]), topic_id, topic["title"]))
                    q_count += 1

    with open(in_file_2, "r") as fh:
        logger.info(f"Importing: {in_file}")
        source = json.load(fh)
        for topic_id, topic in tqdm(enumerate(source["data"])):
            test_count = 0
            for para in topic["paragraphs"]:
                for qas in para['qas']:
                    if test_count < 5:
                        test_count += 1
                        test_new_data["data"].append((quick_clean(raw_str=qas["question"]), topic_id, topic["title"]))
                    else:
                        train_data["data"].append((quick_clean(raw_str=qas["question"]), topic_id, topic["title"]))
                    q_count += 1

    logger.info(f"Saving new data to {out_file}")
    save(filename=train_out_file, obj=train_new_data)
    logger.info(f"Saving new data to {out_file}")
    save(filename=test_out_file, obj=test_new_data)

if __name__ == "__main__":
    args = get_exp3_transformer_args()
    log = get_logger(log_dir=args.logging_dir, name="data-gen")
    exp3_transformer(in_file_1=args.in_file_1, 
                     in_file_2=args.in_file_2,
                     train_out_file=args.train_out_file, 
                     test_out_file=args.test_out_file,
                     logger=log)
