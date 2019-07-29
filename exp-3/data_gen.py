from args import get_exp3_transformer_args
from toolkit import get_logger, quick_clean, save
import ujson as json
from tqdm import tqdm

def exp3_transformer(in_file, out_file, logger):
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
    q_count = 0
    with open(in_file, "r") as fh:
        logger.info(f"Importing: {in_file}")
        source = json.load(fh)
        new_data["version"] = source["version"]
        new_data["data"] = []
        for topic_id, topic in tqdm(enumerate(source["data"])):
            topic_dict = {}
            topic_dict["title"] = topic["title"]
            topic_dict["questions"] = []
            for para in topic["paragraphs"]:
                for qas in para['qas']:
                    topic_dict["questions"].append((quick_clean(raw_str=qas["question"]), topic_id))
                    q_count += 1

    logger.info(f"Saving new data to {out_file}")
    save(filename=out_file, obj=new_data)

if __name__ == "__main__":
    args = get_exp3_transformer_args()
    log = get_logger(log_dir=args.logging_dir, name="data-gen")
    exp3_transformer(in_file=args.in_file, 
                    out_file=args.out_file, 
                    logger=log)
