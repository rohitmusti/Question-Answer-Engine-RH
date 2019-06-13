import ujson as json
from toolkit import fancyprint
import config



def orig_data_discovery(filename):
    """
    just a function to explore the original train data
    """
    tab = " -> "
    with open(filename, "r") as fh:
        source = json.load(fh)
        print("the type of source:", type(source))
        print("the keys of source:",source.keys())
        print("the type of version:",type(source["version"]))
        print("the type of data:",type(source["data"]))
        for article in source["data"]:
            print(tab,"the type of each element of data:",type(article))
            print(tab,"the keys of each element of data:",article.keys())
            print(tab,"the type of title:",type(article["title"]))
            print(tab,"the type of paragraphs:",type(article["paragraphs"]))
            for para in article["paragraphs"]:
                print(tab*2,"the type of each element of paragraphs:",type(para))
                print(tab*2,"the keys of each element of paragraphs:",para.keys())
                print(tab*2,"the type of context:",type(para["context"]))
                print(tab*2,"the type of qas:",type(para["qas"]))
                for qa in para["qas"]:
                    print(tab*3,"the type of each element of qas:",type(qa))
                    print(tab*3,"the keys of each element of qas:",qa.keys())
                    print(tab*3,"the type of id:",type(qa["id"]))
                    print(tab*3,"the type of is_impossible:",type(qa["is_impossible"]))

                    print(tab*3,"the type of question:",type(qa["question"]))
                    print(tab*3,"the type of answers:",type(qa["answers"]))
                    for answer in qa["answers"]:
                        print(tab*4,"the type of each element of answers:",type(answer))
                        print(tab*4,"the keys of each element of answer:",answer.keys())
                        print(tab*4,"the type of text:",type(answer["text"]))
                        print(tab*4,"the type of answer_start:",type(answer["answer_start"]))
                        return None

def exp2_data_discovery(filename):
    """
    just a function to explore some data
    """
    tab = " -> "
    with open(filename, "r") as fh:
        source = json.load(fh)
        print("the type of source:", type(source))
        print("the keys of source:",source.keys())
        print("the type of experiment:",type(source["experiment"]))
        print("the type of version:",type(source["version"]))
        print("the type of data:",type(source["data"]))
        for topic in source["data"]:
            print(tab,"the type of each element of data:",type(topic))
            print(tab,"the keys of each element of data:",topic.keys())
            print(tab,"the type of title:",type(topic["title"]))
            print(tab,"the type of topic_context:",type(topic["topic_context"]))
            print(tab,"the type of qas:",type(topic["qas"]))
            for qas in topic["qas"]:
                print(tab*2,"the type of each element in qas", type(qas))
                print(tab*2,"the keys of each element in qas", qas.keys())
                print(tab*2,"the type of id:",type(qas["id"]))
                print(tab*2,"the type of is_impossible:",type(qas["is_impossible"]))
                print(tab*2,"the type of question:",type(qas["question"]))
                print(tab*2,"the type of answers:",type(qas["answers"]))
                for answer in qas["answers"]:
                    print(tab*3,"the type of each element in answers", type(answer))
                    print(tab*3,"the keys of each element in answers", answer.keys())
                    print(tab*3,"the type of text:",type(answer["text"]))
                    print(tab*3,"the type of answer_start:",type(answer["answer_start"]))

                    return 0

if __name__ == "__main__":
    choice = int(input("select 0 for original or the number corresponding to your experiment: "))
    data = config.data()
    if choice == 0:
        fancyprint(in_str="Original Data")
        orig_data_discovery(filename=data.train_data_orig)
        print()
    elif choice == 2:
        fancyprint(in_str="Experiment 2 Data")
        exp2_data_discovery(filename=data.train_data_exp2)
        print()
    else:
        print("Not implemented yet")