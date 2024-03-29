import argparse

def get_data_gen_args():
    parser = argparse.ArgumentParser("Arguments for generating the data\n NOTE: the data generated by this file is only subset of the training data")
    _add_common_exp2_args(parser)
    parser.add_argument("--train_topic_num",
                        type=int,
                        default=50,
                        help="The number of topics in the original data put in the training dataset")
    parser.add_argument("--dev_topic_num",
                        type=int,
                        default=20,
                        help="The number of topics in the original data put in the dev dataset")
    parser.add_argument("--test_topic_num",
                        type=int,
                        default=10,
                        help="The number of topics in the original data put in the dev dataset")
    args = parser.parse_args()
    return args

def get_exp2_data_transform_args():
    parser = argparse.ArgumentParser("Arguments for transforming raw_data into exp 2 format")
    _add_common_exp2_args(parser)
    args = parser.parse_args()
    return args

def get_exp2_setup_args():
    parser = argparse.ArgumentParser("Arguments for experiment 2 data setup (i.e. building features)")
    _add_common_exp2_args(parser)
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
    args = parser.parse_args()
    return args

def get_exp2_training_args():
    parser = argparse.ArgumentParser("Arguments for running exp2 training")
    _add_common_exp2_args(parser)
    parser.add_argument('--starting_step',
                        type=int,
                        default=0,
                        help='the step to start at')
    parser.add_argument('--starting_epoch',
                        type=int,
                        default=0,
                        help='the epoch to start at')
    parser.add_argument('--starting_train_set',
                        type=int,
                        default=0,
                        help='the epoch to start at')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='The drop probability for the dropout layers.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.5,
                        help='Learning rate.')
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
    parser.add_argument('--learning_rate_decay',
                        type=float,
                        default=0,
                        help='learning_rate_decay.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=10,
                        help='Max number of checkpoints to store.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of subprocesses per data loader.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')

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

def get_exp2_data_gen_args():
    parser = argparse.ArgumentParser("Arguments for sub-sampling the raw data into smaller amounts to work w/ toy datasets")
    _add_common_exp2_args(parser)
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


def _add_common_exp2_args(parser):
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
                        help="The raw data from squad source")
    parser.add_argument("--train_data_src",
                        type=str,
                        default="data/train/train-v2.0.json", 
                        help="The subset of the raw data reserved for training")
    parser.add_argument("--dev_data_src",
                        type=str,
                        default="data/dev/dev-v2.0.json", 
                        help="The subset of the raw data reserved for development (evaluating and hypertuning)")
    parser.add_argument("--test_data_src",
                        type=str,
                        default="data/test/test-v2.0.json", 
                        help="The subset of the raw data reserved for development (evaluating and hypertuning)")
    parser.add_argument("--datasplit",
                        type=str,
                        default="all",
                        choices=("all", "train", "dev", "test"),
                        help="The number of topics in the original data put in the dev dataset")
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='the size of each batch')
    parser.add_argument('--random_seed',
                        type=int,
                        default=3716,
                        help='The default random seed.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help='The standard hidden size for nn layers.')
    parser.add_argument("--train_data_exp2",
                        type=str,
                        default="data/train/train-exp2.json", 
                        help="The subset of the raw data reserved for training")
    parser.add_argument("--dev_data_exp2",
                        type=str,
                        default="data/dev/dev-exp2.json", 
                        help="The subset of the raw data reserved for development (evaluating and hypertuning)")
    parser.add_argument("--test_data_exp2",
                        type=str,
                        default="data/test/test-exp2.json", 
                        help="The subset of the raw data reserved for development (evaluating and hypertuning)")
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
    parser.add_argument("--train_eval_file",
                        type=str,
                        default="data/train/train-eval.json", 
                        help="File reserved for storing train evaluation data")
    parser.add_argument("--dev_eval_file",
                        type=str,
                        default="data/dev/dev-eval.json", 
                        help="File reserved for storing dev evaluation data")
    parser.add_argument("--train_record_file_exp2",
                        type=str,
                        default="data/train/train-exp2", 
                        help="File reserved storing the processed and prepped training data")
    parser.add_argument("--exp2_train_topic_contexts",
                        type=str,
                        default="data/train/exp2_topic_contexts", 
                        help="File reserved storing the processed and prepped topic_contexts")
    parser.add_argument("--exp2_dev_topic_contexts",
                        type=str,
                        default="data/dev/exp2_topic_contexts", 
                        help="File reserved storing the processed and prepped topic_contexts")
    parser.add_argument("--dev_record_file_exp2",
                        type=str,
                        default="data/dev/dev-exp2", 
                        help="File reserved storing the processed and prepped dev data")
    parser.add_argument("--dev_meta_file",
                        type=str,
                        default="data/dev/dev-meta.json", 
                        help="Dev meta file information")
    parser.add_argument('--max_ans_len',
                        type=int,
                        default=15,
                        help='Maximum length of a predicted answer.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--chunk_size',
                         type=int,
                         default=4,
                         help='the number of chunks you want to split the train data into')
    parser.add_argument('--num_train_chunks',
                         type=int,
                         default=None,
                         help='the number of chunks you have to iterate through while training')
    parser.add_argument('--num_dev_chunks',
                         type=int,
                         default=None,
                         help='the number of chunks you have to iterate through while evaluating on dev')
    parser.add_argument('--ques_limit',
                        type=int,
                        default=50,
                        help='question number of words limit.')
    parser.add_argument('--para_limit',
                        type=int,
                        default=4000,
                        help='paragraph character limit.')
    parser.add_argument('--ans_limit',
                        type=int,
                        default=30,
                        help='answer word limit.')
    parser.add_argument('--char_limit',
                        type=int,
                        default=16,
                        help='max number of chars to keep from a word limit.')


