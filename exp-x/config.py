class data():
    """
    borrowd some naming conventions from @chrischute here: https://github.com/chrischute/squad/blob/master/args.py

    author: @rohitmusti
    """
    def __init__(self, folder="./data/"):
        self.folder = folder

        # training data
        # - usually used when you want to "officially" train your model
        self.train_folder = self.folder + "train/"
        self.train_data_orig = self.train_folder + "train-v2.0.json"
        self.train_data_exp2 = self.train_folder + "train-exp2.json"
        self.train_data_exp3 = self.train_folder + "train-exp3.json"
        self.train_eval_exp2 = self.train_folder + "train-eval-exp2.json"
        self.train_eval_exp3 = self.train_folder + "train-eval-exp3.json"
        self.train_record_file_exp2 = self.train_folder + "train-exp2.dev"
        self.train_record_file_exp3 = self.train_folder + "train-exp3.dev"

        # development data
        #  - usually reserved for hypertuning or for training locally to ensure models work
        self.dev_folder = self.folder + "dev/"
        self.dev_data_orig = self.dev_folder + "dev-v2.0.json"
        self.dev_data_exp2 = self.dev_folder + "dev-exp2.json"
        self.dev_data_exp3 = self.dev_folder + "dev-exp3.json"
        self.dev_eval_exp2 = self.dev_folder + "dev-eval-exp2.json"
        self.dev_eval_exp3 = self.dev_folder + "dev-eval-exp3.json"
        self.dev_record_file_exp2 = self.dev_folder + "dev-exp2.npz"
        self.dev_record_file_exp3 = self.dev_folder + "dev-exp3.npz"

        # toy data
        # -  just to see if things are working
        self.toy_folder = self.folder + "toy/"
        self.toy_data_orig = self.toy_folder + "toy-v2.0.json"
        self.toy_data_exp2 = self.toy_folder + "toy-exp2.json"
        self.toy_data_exp3 = self.toy_folder + "toy-exp3.json"
        self.toy_eval_exp2 = self.toy_folder + "toy-eval-exp2.json"
        self.toy_eval_exp3 = self.toy_folder + "toy-eval-exp3.json"
        self.toy_record_file_exp2 = self.toy_folder + "toy-exp2.npz"
        self.toy_record_file_exp3 = self.toy_folder + "toy-exp3.npz"
        self.toy_word2idx_file = self.toy_folder + "toy_word2idx.json"
        self.toy_char2idx_file = self.toy_folder + "toy_char2idx.json"
        self.toy_word_emb_file = self.toy_folder + "toy_word_emb.json"
        self.toy_char_emb_file = self.toy_folder + "toy_char_emb.json"

        # word embeddings
        self.embed_folder = self.folder + "embeddings/"
        self.glove_word_file = self.embed_folder + "glove.840B.300d.txt"
        self.glove_word_dim = 300
        self.glove_word_num_vecs = 2196017
        self.glove_char_file = self.embed_folder + "glove.840B.300d-char.txt"
        self.char_emb_size = 64
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

        self.hidden_size = 100
        self.drop_prob = 0.2

        # save file to save logs
        self.logging_dir = "./logs/"
        self.load_path = None