import ujson as json
from toolkit import get_logger, quick_clean, save
from random import randrange
from tqdm import tqdm
from args import get_exp1_transform_args

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
        logger.info(f"Importing {fh.name}")
        source = json.load(fh)
        new_data["version"] = source["version"]
        new_data["data"] = []
        logger.info("Creating all context list")
        all_contexts = [para["context"] for topic in source["data"] for para in topic["paragraphs"]]
        for topic in tqdm(source["data"]):
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

    logger.info(f"Processed {counter} question, answer pairs")
    logger.info(f"Saving to {out_file}")
    save(filename=out_file, obj=new_data)

if __name__ == "__main__":
    args = get_exp1_transform_args()
    logger = get_logger(log_dir=args.logging_dir, name="exp_1 data transformer")

    # standard sanity check to run every time
    c, b = get_new_context("test", ["test1", "test2", "test3", "test4", "test5",
                           "test6", "test7", "test8", "test9"])
    test_val = "test" == c[b:b+4]
    if test_val != True:
        raise ValueError('The get_new_context function is not working')
    
    if args.datasplit=="train" or args.datasplit=="all":
        exp_1_transformer(args.train_data_src, args.train_data_exp1, logger)
    if args.datasplit=="dev" or args.datasplit=="all":
        exp_1_transformer(args.dev_data_src, args.dev_data_exp1, logger)
    if args.datasplit=="test" or args.datasplit=="all":
        exp_1_transformer(args.test_data_src, args.test_data_exp1, logger)
