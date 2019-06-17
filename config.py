class data():
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
        # development data
        #  - usually reserved for hypertuning or for training locally to ensure models work
        self.dev_folder = self.folder + "dev/"
        self.dev_data_orig = self.dev_folder + "dev-v2.0.json"
        self.dev_data_exp2 = self.dev_folder + "dev-exp2.json"
        self.dev_data_exp3 = self.dev_folder + "dev-exp3.json"
        # word embeddings
        self.embed_folder = self.folder + "embeddings/"
        self.word_embeddings = self.embed_folder + "glove.840B.300d.txt"
        self.char_embeddings = self.embed_folder + "glove.840B.300d-char.txt"