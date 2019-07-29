import ujson as json
from tqdm import tqdm

def longest_question(data_src):
    max_len = 0
    longest_question = ""
    long_tid = ""
    long_tt = ""
    with open(data_src, "r") as fh:
        source = json.load(fh)
        for question, topic_id, topic_title in tqdm(source['data']):
            if len(question) > max_len:
                max_len = len(question)
                longest_question = question
                long_tid = topic_id
                long_tt = topic_title
        print(f"longest_question: {longest_question}")
        print(f"length: {max_len}")
        print(f"from: #{long_tid} {long_tt}")

def average_question(data_src):
    longest_question = ""
    with open(data_src, "r") as fh:
        source = json.load(fh)
        question_count = len(source["data"])
        question_lens = [len(question) for question, _, _ in source['data']]
        all_lens = sum(question_lens)
        average_len = all_lens / question_count
        print(f"average_question_length: {average_len}")
            

if __name__ == "__main__":
    train_src = "./data/clean-train-exp3.json"
    dev_src = "./data/clean-dev-exp3.json"
    print("############## train results")
    longest_question(train_src)
    average_question(train_src)
    print("############## dev results")
    longest_question(dev_src)
    average_question(dev_src)
    
