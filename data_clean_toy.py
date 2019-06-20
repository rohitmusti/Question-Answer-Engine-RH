"""
tldr: modifies the squad data to fit the requirements of my experiment 2 (topic contexts)

author: @rohitmusti
"""
import ujson as json
from tqdm import tqdm
from toolkit import fancyprint, save, quick_clean
import config
from data_clean_exp2 import exp2_transformer
from data_clean_exp3 import exp3_transformer

def toy_transformer(in_file, out_file):
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
    with open(in_file, "r") as fh:
        fancyprint(in_str=("Importing: " + in_file))
        source = json.load(fh)
        fancyprint(in_str="Converting into toy format")
        new_data["version"] = source["version"]
        new_data["data"] = []
        topic_counter = 15
        for topic in tqdm(source["data"]):
            topic_dict = {}
            topic_dict["title"] = topic["title"]
            topic_dict["paragraphs"] = []
            para_counter = 5
            for para in topic["paragraphs"]:
                paragraph = {}
                paragraph["context"] = para["context"]
                paragraph["qas"] = []
                qa_counter = 5
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

                    qa_counter -= 1
                    if qa_counter == 0:
                        break

                topic_dict["paragraphs"].append(paragraph)
                para_counter -= 1
                if para_counter == 0:
                    break

            new_data["data"].append(topic_dict)

            topic_counter -= 1
            if topic_counter == 0:
                break

    save(filename=out_file, obj=new_data, message="saving toy data")



if __name__ == "__main__":
    data = config.data()
    toy_transformer(in_file=data.train_data_orig, out_file=data.toy_data_orig)
    exp2_transformer(in_file=data.toy_data_orig, out_file=data.toy_data_exp2)
    exp3_transformer(in_file=data.toy_data_orig, out_file=data.toy_data_exp3)
