import ujson as json

def fancyprint(in_str):
    print()
    print("#"*20)
    print("# " + in_str)
    print("#"*20)
    print()

def data_discovery(filename):
    """
    just a function to explore some data
    """
    tab = " -> "
    with open(filename, "r") as fh:
        source = json.load(fh)
        print("the type of source:", type(source))
        print("the keys of source:",source.keys())
        print("the type of version:",type(source['version']))
        print("the type of data:",type(source['data']))
        for article in source["data"]:
            print(tab,"the type of each element of data:",type(article))
            print(tab,"the keys of each element of data:",article.keys())
            print(tab,"the type of title:",type(article['title']))
            print(tab,"the type of paragraphs:",type(article['paragraphs']))
            for para in article["paragraphs"]:
                print(tab*2,"the type of each element of paragraphs:",type(para))
                print(tab*2,"the keys of each element of paragraphs:",para.keys())
                print(tab*2,"the type of context:",type(para['context']))
                print(tab*2,"the type of qas:",type(para['qas']))
                for qa in para["qas"]:
                    print(tab*3,"the type of each element of qas:",type(qa))
                    print(tab*3,"the keys of each element of qas:",qa.keys())
                    print(tab*3,"the type of id:",type(qa['id']))
                    print(tab*3,"the type of is_impossible:",type(qa['is_impossible']))
                    print(tab*3,"the type of question:",type(qa['question']))
                    print(tab*3,"the type of answers:",type(qa['answers']))
                    for answer in qa["answers"]:
                        print(tab*4,"the type of each element of answers:",type(answer))
                        print(tab*4,"the keys of each element of answer:",answer.keys())
                        print(tab*4,"the type of text:",type(answer['text']))
                        print(tab*4,"the type of answer_start:",type(answer['answer_start']))
                        return None


if __name__ == '__main__':
    fancyprint(in_str="Original Data")
    data_discovery(filename="./data/train/train-v2.0.json")