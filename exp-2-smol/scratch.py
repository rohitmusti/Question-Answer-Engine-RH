import ujson as json
import matplotlib.pyplot as plt

def explorer(in_file):
    contexts = {}
    lengths = []
    lengths2 = []
    counter = 0
    with open(in_file, "r") as fh:
        source = json.load(fh)
        for topic_id, topic in (enumerate(source["data"])):
            topic_contexts = [para["context"] for para in topic["paragraphs"]]
            if len(topic_contexts) < 100:
                # print('boo')
                counter += 1
            topic_context = " ".join(topic_contexts)
            lengths.append(len(topic_context))
            lengths2.append(len(topic_context.split()))
            contexts[topic['title']] = topic_context
        print(f"num smol bois {counter}")

    # plt.hist(lengths, bins='auto')
    # plt.title("Histogram of num chars")
    # plt.show()

    # plt.hist(lengths2, bins='auto')
    # plt.title("Histogram of num words")
    # plt.show()
    sub = []
    for i in lengths2:
        if i < 8000:
            sub.append(i)
    print(f"Num less than 8000 words {len(sub)}/{len(lengths2)}")

    max_tracker = 0
    max_tracker2 = 0
    max_cand = ""
    max_cand2 = ""
    summer = 0
    summer2 = 0
    for key in contexts:
        if len(contexts[key].split()) > max_tracker:
            max_tracker = len(contexts[key].split())
            max_cand = key
        summer += len(contexts[key].split())
        if len(contexts[key]) > max_tracker:
            max_tracker2 = len(contexts[key])
            max_cand2 = key
        summer2 += len(contexts[key])

    print(f"max num words: {max_tracker}")
    print(f"title: {max_cand}")
    print(f"average: {summer/len(contexts)}")
    print("#####################")
    print(f"max num chars: {max_tracker2}")
    print(f"title: {max_cand2}")
    print(f"average: {summer2/len(contexts)}")



if __name__ == "__main__":
    print("train")
    explorer("./data/train/orig-train-v2.0.json")
    print()
    print("dev")
    explorer("./data/dev/orig-dev-v2.0.json")
