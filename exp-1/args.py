import argparse

def get_data_gen_args():
    parser = argparse.ArgumentParser("Arguments for sub-sampling the raw data into smaller amounts to work w/ toy datasets")
    _add_common_exp1_args(parser)
    args = parser.parse_args()
    return args

def _add_common_exp1_args(parser):
    parser.add_argument("--logging_dir",
                        type=str,
                        default="./logs/",
                        help="The folder where all logs are stored")
    parser.add_argument("--save_dir",
                        type=str,
                        default="./save/",
                        help="The folder for storing general info")
    parser.add_argument("--load_path",
                        type=str,
                        default=None,
                        help="Load path where old good models are stored")
    parser.add_argument("--raw_train_data",
                        type=str,
                        default="data/train/orig-train-v2.0.json", 
                        help="The entire corpus of raw training data")
    parser.add_argument("--raw_dev_data",
                        type=str,
                        default="data/dev/orig-dev-v2.0.json", 
                        help="The entire corpus of raw dev data")
    parser.add_argument("--raw_test_data",
                        type=str,
                        default="data/test/orig-test-v2.0.json", 
                        help="The entire corpus of raw test data")
    parser.add_argument("--train_data_src",
                        type=str,
                        default="data/train/train-v2.0.json", 
                        help="The subset of the raw data used as the training source")
    parser.add_argument("--dev_data_src",
                        type=str,
                        default="data/dev/dev-v2.0.json", 
                        help="The subset of the raw data used as the development sourcee")
    parser.add_argument("--test_data_src",
                        type=str,
                        default="data/test/test-v2.0.json", 
                        help="The subset of the raw data used as the  testing source")
    parser.add_argument("--datasplit",
                        type=str,
                        default="all",
                        choices=("all", "train", "dev", "test"),
                        help="The number of topics in the original data put in the dev dataset")