import numpy as np
import pandas as pd
import json as json
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import os

from args import get_exp3_train_args

def make_feature_vec(words, num_features):
    """
    Average the word vectors for a set of words
    """
    feature_vec = np.zeros((300,),dtype="float32")  # pre-initialize (for speed)
    nwords = 0.

    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec,model[word])
    
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

def main(args):
    t_raw = np.load(args.train_feature_file)
    d_raw = np.load(args.dev_feature_file)

    with open(args.word_emb_file, 'r') as fh:
        word_vectors = (json.load(fh))
    
    print('retrieving embeddings for train...')
    t_x = []
    for word_array in t_raw["qw_idxs"]:
        nwa = np.zeros((300,),dtype="float32")  # pre-initialize (for speed)
        for word in word_array:
            nwa = np.add(nwa, word_vectors[word])
        # nwa = np.divide(nwa, len(word_array))
        t_x.append(np.asarray(nwa))

    print('retrieving embeddings for dev...')
    d_x = []
    for word_array in d_raw["qw_idxs"]:
        nwa = np.zeros((300,),dtype="float32")  # pre-initialize (for speed)
        for word in word_array:
            nwa = np.add(nwa, word_vectors[word])
        # nwa = np.divide(nwa, len(word_array))
        d_x.append(np.asarray(nwa))


    print('creating dataframes...')
    t_X = pd.DataFrame(t_x,
                      columns=[str(i+1) for i in range(300)])
    d_X = pd.DataFrame(d_x,
                      columns=[str(i+1) for i in range(300)])
    t_Y = pd.DataFrame(t_raw['topic_ids'], columns=['topic_ids'])
    d_Y = pd.DataFrame(d_raw['topic_ids'], columns=['topic_ids'])

    print('training...')
    forest = RandomForestClassifier(n_estimators=1000, n_jobs=os.cpu_count(), verbose=True)
    forest = forest.fit(t_X, t_raw['topic_ids'])
    predictions = forest.predict(d_X)
    prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')


    
    



if __name__ == "__main__":
    args = get_exp3_train_args()
    main(args)

