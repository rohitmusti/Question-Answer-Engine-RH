import ujson as json
from toolkit import get_logger, quick_clean, save
from config import config
from args import get_exp2_data_transform_args
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
            context_buffer = 0
            topic_dict = {}
            topic_dict["title"] = topic["title"]
            topic_dict["qas"] = []
            topic_contexts = [para["context"] for para in topic["paragraphs"]]
            topic_contexts = " ".join(topic_contexts)
            if len(topic_contexts.split()) < 8000:
                topic_dict["context"] = 
            else:
                continue
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
    args = get_exp2_data_transform_args()
    datasplit = args.datasplit
    logger = get_logger(log_dir=args.logging_dir, name="exp2 data transformer")

    if datasplit=="train" or datasplit=="all":
        exp2_transformer(args.train_data_src, args.train_data_exp2, logger)
    if datasplit=="dev" or datasplit=="all":
        exp2_transformer(args.dev_data_src, args.dev_data_exp2, logger)
    if datasplit=="test" or datasplit=="all":
        exp2_transformer(args.test_data_src, args.test_data_exp2, logger)
    else:
        raise ValueError("Unrecognized or missing flags")
