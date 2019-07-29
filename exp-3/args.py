import argparse

def get_exp3_featurize_args():
    parser = argparse.ArgumentParser("Arguments for featurizing data in prep for training")
    _add_common_exp3_args(parser)

    parser.add_argument("--ques_limit",
                        type=int,
                        default=50,
                        help="The max number of words to keep from a question.")
    parser.add_argument("--glove_file",
                        type=str,
                        default="./data/glove.840B.300d.txt",
                        help="The file containing the word vectors.")
    parser.add_argument("--word_emb_file",
                        type=str,
                        default="./data/word-emb.json",
                        help="The file containing word embedding maps.")
    parser.add_argument("--train_in_file",
                        type=str,
                        default="./data/clean-train-exp3.json",
                        help="The file containing the cleand and shrunken train data.")
    parser.add_argument("--dev_in_file",
                        type=str,
                        default="./data/clean-dev-exp3.json",
                        help="The file containing the cleand and shrunken dev data.")
    parser.add_argument("--train_feature_file",
                        type=str,
                        default="./data/train-prepped-exp3.npz",
                        help="The file containing the featurized train data.")
    parser.add_argument("--dev_feature_file",
                        type=str,
                        default="./data/dev-prepped-exp3.npz",
                        help="The file containing the featurized dev data.")
    parser.add_argument("--train_topic_title_id_map_file",
                        type=str,
                        default="./data/train-topic-title-toppic-id-map.json",
                        help="The file containing the mapping from topic_id to topic for the training data.")
    parser.add_argument("--dev_topic_title_id_map_file",
                        type=str,
                        default="./data/dev-topic-title-toppic-id-map.json",
                        help="The file containing the mapping from topic_id to topic for the dev data.")
    parser.add_argument("--train_eval_file",
                        type=str,
                        default="./data/train-eval.json",
                        help="The file containing data required to evaluate on the train set.")
    parser.add_argument("--dev_eval_file",
                        type=str,
                        default="./data/dev-eval.json",
                        help="The file containing data required to evaluate on the dev set.")
    
    args = parser.parse_args()
    return args

def get_exp3_transformer_args():
    parser = argparse.ArgumentParser("Arguments for transforming data into a useful form for classification")
    _add_common_exp3_args(parser)
    parser.add_argument("--in_file",
                        type=str,
                        default="./data/train-v2.0.json",
                        help="The input file of raw data to be transformed. Designed for SQuAD v2.0")
    parser.add_argument("--out_file",
                        type=str,
                        default="./data/clean-train-exp3.json",
                        help="The output file of where the data set up for exp3 is stored. Ready to be featurized.")
    args = parser.parse_args()
    return args


def _add_common_exp3_args(parser):
    parser.add_argument("--logging_dir",
                        type=str,
                        default="./logs/",
                        help="The folder where all logs are stored")
    parser.add_argument("--data_split",
                        type=str,
                        default="train",
                        choices=("train", "test"),
                        help="The folder where all logs are stored")
