import argparse

def get_exp3_featurize_args():
    parser = argparse.ArgumentParser("Arguments for featurizing data in prep for training")
    _add_common_exp3_args(parser)

    parser.add_argument("--ques_limit",
                        type=int,
                        default=40,
                        help="The max number of words to keep from a question.")
    parser.add_argument("--glove_file",
                        type=str,
                        default="./data/glove.840.300d.txt",
                        help="The file containing the word vectors.")
    
    args = parser.parse_args()
    args.train_in_file = "./data/clean-train-exp3.json"
    args.train_feature_file = "./data/train-prepped-exp3.npz"
    args.train_topic_title_id_map_file = "./data/train-title-topic-id-map.json"
    args.train_eval_examples_file = "./data/train-eval-examples.json"
    args.train_feature_file = "./data/train-features.npz"
    args.dev_in_file = "./data/clean-dev-exp3.json"
    args.dev_feature_file = "./data/dev-prepped-exp3.npz"
    args.dev_topic_title_id_map_file = "./data/dev-title-topic-id-map.json"
    args.dev_eval_examples_file = "./data/dev-eval-examples.json"

    args.word_emb_file = "./data/word-emb.json"
    else:
        raise ValueError(f'Unrecognized metric name: "{args.data_split}"')

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
