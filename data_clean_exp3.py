"""
tldr: modifies the squad data to fit the requirements of my experiment 2 (topic contexts)

author: @rohitmusti
"""
import ujson as json
from tqdm import tqdm
from toolkit import fancyprint, save, quick_clean
import config


def exp3_transformer(in_file, out_file):
    """
    args:
        - in_file: the file name of the data to be transformed to experiment 2
        - out_file: the file name of where the ought to be written

    return:
        none, the data is written to an output
    """
    new_data = {}
    new_data['experiment'] = 3
    with open(in_file, "r") as fh:
        fancyprint(in_str=("Importing: " + in_file))
        source = json.load(fh)
        fancyprint(in_str="Converting into experiment 3 format")
        new_data["version"] = source["version"]
        new_data["data"] = []
        # Merge all the `contexts` into one giant string
        new_data["super_context"] = "".join([quick_clean(raw_str=para["context"]) for topic in source["data"] for para in topic["paragraphs"]])
        context_buffer = 0
        for topic in tqdm(source["data"]):
            topic_dict = {}
            topic_dict["title"] = topic["title"]
            topic_dict["qas"] = []
            for para in topic["paragraphs"]:
                for qas in para['qas']:
                    qas_dict = {}
                    qas_dict["id"] = qas["id"]
                    qas_dict["is_impossible"] = qas["is_impossible"]
                    qas_dict["question"] = quick_clean(raw_str=qas["question"])
                    qas_dict["answers"] = []
                    if not qas["is_impossible"]:
                        for answer in qas["answers"]:
                            answer_dict = {}
                            # update the answer start index
                            answer_dict["answer_start"] = answer["answer_start"] + context_buffer
                            answer_dict["text"] = answer["text"]
                            qas_dict["answers"].append(answer_dict)
                    topic_dict["qas"].append(qas_dict)
                context_buffer += len(para["context"])
            new_data["data"].append(topic_dict)

    save(filename=out_file, obj=new_data, message="saving experiment 3 data")



if __name__ == "__main__":
    data = config.data()
    exp3_transformer(in_file=data.train_data_orig, out_file=data.train_data_exp3)
    exp3_transformer(in_file=data.dev_data_orig, out_file=data.dev_data_exp3)
