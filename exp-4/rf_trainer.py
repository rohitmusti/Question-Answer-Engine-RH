import numpy as np
import pandas as pd
import ujson as json
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from args import get_exp3_train_args

def main(args):
    t_raw = np.load(args.train_feature_file)
    d_raw = np.load(args.dev_feature_file)

    with open(args.word_emb_file, 'r') as fh:
        word_vectors = (json.load(fh))
    
    print('retrieving embeddings for train...')
    t_x = []
    for word_array in t_raw["qw_idxs"]:
        nwa = []
        for word in word_array:
            nwa.append(word_vectors[word])
        t_x.append(np.asarray(nwa))

    print('retrieving embeddings for dev...')
    d_x = []
    for word_array in d_raw["qw_idxs"]:
        nwa = []
        for word in word_array:
            nwa.append(word_vectors[word])
        d_x.append(np.asarray(nwa))


    print('creating dataframes...')
    t_X = pd.DataFrame(t_x,
                      columns=[str(i+1) for i in range(31)])
    d_X = pd.DataFrame(np.array(d_raw['qw_idxs']),
                      columns=[str(i+1) for i in range(31)])
    t_Y = pd.DataFrame(t_raw['topic_ids'], columns=['topic_ids'])
    d_Y = pd.DataFrame(d_raw['topic_ids'], columns=['topic_ids'])

    print('training...')
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit(t_X, t_Y)

    
    



if __name__ == "__main__":
    args = get_exp3_train_args()
    main(args)

