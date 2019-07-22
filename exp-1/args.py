import argparse

def get_exp1_train_test_args():
    parser = argparse.ArgumentParser("Arguments for exp1 training. Reminder to set the load_path for this to work.")
    _add_common_exp1_args(parser)
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='the size of each batch')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of subprocesses per data loader.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--random_seed',
                        type=int,
                        default=3716,
                        help='The default random seed.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help='The standard hidden size for nn layers.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='The drop probability for the dropout layers.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=10,
                        help='Max number of checkpoints to store.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.5,
                        help='Learning rate.')
    parser.add_argument('--learning_rate_decay',
                        type=float,
                        default=0,
                        help='learning_rate_decay.')
    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between evaluations.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')

    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')


    return args

def get_exp1_setup_args():
    parser = argparse.ArgumentParser("Arguments for transforming the raw data into exp1 format")
    _add_common_exp1_args(parser)
    parser.add_argument("--glove_word_file",
                        type=str,
                        default="data/embeddings/glove.840B.300d.txt", 
                        help="Glove word file")
    parser.add_argument("--glove_word_dim",
                        type=int,
                        default=300,
                        help="Glove word dimensions")
    parser.add_argument("--glove_word_num_vecs",
                        type=int,
                        default=2196017,
                        help="Glove word dimensions")
    parser.add_argument("--glove_char_file",
                        type=str,
                        default="data/embeddings/glove.840B.300d-char.txt", 
                        help="Glove char file")
    parser.add_argument("--glove_char_dim",
                        type=int,
                        default=64,
                        help="Glove char dimension")
    parser.add_argument('--file_batch_size',
                        type=int,
                        default=18,
                        help='This is the default batch size for multi-processing files during setup')
    args = parser.parse_args()
    return args

def get_exp1_transform_args():
    parser = argparse.ArgumentParser("Arguments for transforming the raw data into exp1 format")
    _add_common_exp1_args(parser)
    args = parser.parse_args()
    return args

def get_exp1_data_gen_args():
    parser = argparse.ArgumentParser("Arguments for sub-sampling the raw data into smaller amounts to work w/ toy datasets")
    _add_common_exp1_args(parser)
    parser.add_argument("--train_topic_num",
                        type=int,
                        default=50,
                        help="The number of topics to put into the training_data_src")
    parser.add_argument("--dev_topic_num",
                        type=int,
                        default=20,
                        help="The number of topics to put into the dev_data_src")
    parser.add_argument("--test_topic_num",
                        type=int,
                        default=10,
                        help="The number of topics to put into the test_data_src")
    args = parser.parse_args()
    return args

def _add_common_exp1_args(parser):
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
    parser.add_argument("--train_data_exp1",
                        type=str,
                        default="data/train/train-exp1.json", 
                        help="The training data in exp1 format")
    parser.add_argument("--dev_data_exp1",
                        type=str,
                        default="data/dev/dev-exp1.json", 
                        help="The dev data in exp1 format")
    parser.add_argument("--test_data_exp1",
                        type=str,
                        default="data/test/test-exp1.json", 
                        help="The test data in exp1 format")
    parser.add_argument("--train_eval_file",
                        type=str,
                        default="data/train/train-eval.json", 
                        help="File reserved for storing train evaluation data")
    parser.add_argument("--dev_eval_file",
                        type=str,
                        default="data/dev/dev-eval.json", 
                        help="File reserved for storing dev evaluation data")
    parser.add_argument("--test_eval_file",
                        type=str,
                        default="data/test/test-eval.json", 
                        help="File reserved for storing test evaluation data")
    parser.add_argument("--dev_meta_file",
                        type=str,
                        default="data/dev/dev-meta.json", 
                        help="File reserved storing the dev meta data")
    parser.add_argument("--train_record_file_exp1",
                        type=str,
                        default="data/train/train-exp1.npz", 
                        help="File reserved storing the processed and prepped training data")
    parser.add_argument("--dev_record_file_exp1",
                        type=str,
                        default="data/dev/dev-exp1.npz", 
                        help="File reserved storing the processed and prepped dev data")
    parser.add_argument("--test_record_file_exp1",
                        type=str,
                        default="data/test/test-exp1.npz", 
                        help="File reserved storing the processed and prepped test data")
    parser.add_argument("--word2idx_file",
                        type=str,
                        default="data/embeddings/word2idx.json", 
                        help="File reserved for storing word2idx pairings")
    parser.add_argument("--char2idx_file",
                        type=str,
                        default="data/embeddings/char2idx.json", 
                        help="File reserved for storing char2idx pairings")
    parser.add_argument("--word_emb_file",
                        type=str,
                        default="data/embeddings/word-emb.json", 
                        help="File reserved for storing word embeddings")
    parser.add_argument("--char_emb_file",
                        type=str,
                        default="data/embeddings/char-emb.json", 
                        help="File reserved for storing character embeddings")
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
    parser.add_argument("--datasplit",
                        type=str,
                        default="all",
                        choices=("all", "train", "dev", "test"),
                        help="The number of topics in the original data put in the dev dataset")
    parser.add_argument('--ques_limit',
                        type=int,
                        default=50,
                        help='question number of words limit.')
    parser.add_argument('--para_limit',
                        type=int,
                        default=1500,
                        help='paragraph character limit.')
    parser.add_argument('--ans_limit',
                        type=int,
                        default=50,
                        help='answer word limit.')
    parser.add_argument('--char_limit',
                        type=int,
                        default=30,
                        help='max number of chars to keep from a word limit.')
