"""
Full of tests to make sure the data manipulation goes correctly.
"""
from toolkit import fancyprint
import ujson as json
from tqdm import tqdm
import config

def test_exp2_data_transform(orig_data, transformed_data):
    fancyprint("Testing Experiment 2 Data Transformation")
    with open(orig_data, "r") as od, open(transformed_data) as td:
        orig_source = json.load(od)
        exp2_source = json.load(td)

        fancyprint("Testing that this is the correct experiment")
        assert exp2_source["experiment"] == 2, "the version should be two"

        fancyprint("Testing that the context lengths add up")

        orig_len, exp2_len = 0, 0
        for article in tqdm(orig_source["data"]):
            for para in article["paragraphs"]:
                orig_len += len(para["context"])
    
        for topic in tqdm(exp2_source["data"]):
            exp2_len += len(topic["topic_context"])
    
        assert orig_len == exp2_len, "the summed context lengths should line up"

        fancyprint("Testing that there are the same number of answers")

        orig_ans_count, exp2_ans_count = 0, 0

        for article in tqdm(orig_source["data"]):
            for para in article["paragraphs"]:
                for qas in para["qas"]:
                    for answer in qas["answers"]:
                        orig_ans_count += 1
    
        for topic in tqdm(exp2_source["data"]):
            for qas in topic["qas"]:
                for answer in qas["answers"]:
                    exp2_ans_count += 1
    
        assert orig_ans_count == exp2_ans_count, "the answer counts should line up"
    

def test_exp3_data_transform(orig_data, transformed_data):
    fancyprint("Testing Experiment 3 Data Transformation")
    with open(orig_data, "r") as od, open(transformed_data) as td:
        orig_source = json.load(od)
        exp3_source = json.load(td)

        fancyprint("Testing that this is the correct experiment")
        assert exp3_source["experiment"] == 3, "the version should be two"

        fancyprint("Testing that the context lengths add up")

        orig_len, exp3_len = 0, 0
        for article in tqdm(orig_source["data"]):
            for para in article["paragraphs"]:
                orig_len += len(para["context"])
    
        assert orig_len == len(exp3_source["super_context"]), "the summed context lengths should line up"

        fancyprint("Testing that there are the same number of answers")

        orig_ans_count, exp3_ans_count = 0, 0

        for article in tqdm(orig_source["data"]):
            for para in article["paragraphs"]:
                for qas in para["qas"]:
                    for answer in qas["answers"]:
                        orig_ans_count += 1
    
        for topic in tqdm(exp3_source["data"]):
            for qas in topic["qas"]:
                for answer in qas["answers"]:
                    exp3_ans_count += 1
    
        assert orig_ans_count == exp3_ans_count, "the answer counts should line up"

if __name__ == "__main__":
    data = config.data()
    test_exp2_data_transform(orig_data=data.train_data_orig, transformed_data=data.train_data_exp2)
    test_exp3_data_transform(orig_data=data.train_data_orig, transformed_data=data.train_data_exp3)
    print()
