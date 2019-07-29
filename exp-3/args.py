import argparse

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
