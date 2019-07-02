class config():
    """
    borrowd some naming conventions from @chrischute here: https://github.com/chrischute/squad/blob/master/args.py
    author: @rohitmusti
    """
    def __init__(self, folder="./data/"):
        self.folder = folder
        self.logging_dir = "./logs/"
        self.save_dir = "./save/"
        self.load_path = None
        self.checkpoints = "./checkpoints/"
        self.name = None
        self.split=None

        # training data
        # - usually used when you want to "officially" train your model
        self.train_folder = self.folder + "train/"
        self.train_data_orig = self.train_folder + "train-v2.0.json"
        self.train_data_exp2 = self.train_folder + "train-exp2.json"
        self.train_eval_file = self.train_folder + "train-eval.json"
        self.train_topic_contexts_file = self.train_folder + "train-topic-contexts.json"
        self.train_record_file_exp2 = self.train_folder + "train-exp2.npz"

        # development data
        #  - usually reserved for hypertuning or for training locally to ensure models work
        self.dev_folder = self.folder + "dev/"
        self.dev_data_orig = self.dev_folder + "dev-v2.0.json"
        self.dev_data_exp2 = self.dev_folder + "dev-exp2.json"
        self.dev_eval_file = self.dev_folder + "dev-eval.json"
        self.dev_topic_contexts_file = self.dev_folder + "dev-topic-contexts.json"
        self.dev_record_file_exp2 = self.dev_folder + "dev-exp2.npz"
        self.dev_meta_file = self.dev_folder + "dev-meta.json"

        # toy data
        # -  just to see if things are working
        self.toy_folder = self.folder + "toy/"
        self.toy_data_orig = self.toy_folder + "toy-v2.0.json"
        self.toy_data_exp2 = self.toy_folder + "toy-exp2.json"
        self.toy_eval_file = self.toy_folder + "toy-eval.json"
        self.toy_topic_contexts_file = self.toy_folder + "toy-topic-contexts.json"
        self.toy_meta_file = self.toy_folder + "toy-meta.json"
        self.toy_record_file_exp2 = self.toy_folder + "toy-exp2.npz"
        self.toy_word2idx_file = self.toy_folder + "toy-word2idx.json"
        self.toy_char2idx_file = self.toy_folder + "toy-char2idx.json"
        self.toy_word_emb_file = self.toy_folder + "toy-word-emb.json"
        self.toy_char_emb_file = self.toy_folder + "toy-char-emb.json"
        self.toy_dev_data_orig = self.toy_folder + "toy-dev-v2.0.json"
        self.toy_dev_data_exp2 = self.toy_folder + "toy-dev-exp2.json"
        self.toy_dev_eval_file = self.toy_folder + "toy-dev-eval.json"
        self.toy_dev_topic_contexts_file = self.toy_folder + "toy-dev-topic-contexts.json"
        self.toy_dev_record_file_exp2 = self.toy_folder + "toy-dev-exp2.npz"
        self.toy_dev_meta_file = self.toy_folder + "toy-dev-meta.json"

        # word embeddings
        self.embed_folder = self.folder + "embeddings/"
        self.glove_word_file = self.embed_folder + "glove.840B.300d.txt"
        self.glove_word_dim = 300
        self.glove_word_num_vecs = 2196017
        self.glove_char_file = self.embed_folder + "glove.840B.300d-char.txt"
        self.char_dim = 64
        self.word2idx_file = self.embed_folder + "word2idx.json"
        self.char2idx_file = self.embed_folder + "char2idx.json"
        self.word_emb_file = self.embed_folder + "word_emb.json"
        self.char_emb_file = self.embed_folder + "char_emb.json"

        # random seed for consistent runs
        self.random_seed = 3716
        self.gpu_ids = []
        self.batch_size = 64
        self.ema_decay = 0.999
        self.metric_name = "F1"
        self.maximize_metric = True
        self.learning_rate = 0.2
        self.learning_weight_decay = 0
        self.num_workers = 1
        self.eval_steps = 50000
        self.num_epochs = 30
        self.max_grad_norm = 5.0
        self.num_visuals = 10
        self.para_limit = 3800
        self.ques_limit = 50
        self.ans_limit = 30
        self.char_limit = 16

        self.max_checkpoints = 5
        self.max_ans_len = 25
        self.metric_name = "F1"
        if self.metric_name == 'NLL':
            # Best checkpoint is the one that minimizes negative log-likelihood
            self.maximize_metric = False
        elif self.metric_name in ('EM', 'F1'):
            # Best checkpoint is the one that maximizes EM or F1
            self.maximize_metric = True
        else:
            raise ValueError(f'Unrecognized metric name: "{self.metric_name}"')

        self.hidden_size = 100
        self.drop_prob = 0.2

        # save file to save logs