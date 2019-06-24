import ujson as json
from config import config

def exp_1_transformer(in_file, out_file):
    print('in file: {}'.format(in_file))
    print('out file: {}'.format(out_file))


if __name__ == "__main__":
    c = config()

    exp_1_transformer(c.train_data_orig, c.train_data_exp1)
    exp_1_transformer(c.dev_data_orig, c.dev_data_exp1)
    exp_1_transformer(c.toy_data_orig, c.toy_data_exp1)
