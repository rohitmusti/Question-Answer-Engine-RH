import ujson as json
import numpy as np
import torch
import torch.utils.data as data

def fancyprint(in_str):
    print()
    print("#"*20)
    print("# " + in_str)
    print("#"*20)
    print()

def save(filename, obj, message=None):
    """
    just saves the file, nothing fancy
    author: @wzhouad
    """
    if message is not None:
        fancyprint("Saving {}!".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

def quick_clean(raw_str):
    """
    args:
        - context: a string to be quickly cleaned

    return
        - the original string w/ all quotes replaced as double quotes
    """
    return raw_str.replace("''", '" ').replace("``", '" ')

class SQuAD(data.Dataset):
    def __init__(self, data_path):
        super(SQuAD, self).__init__()

        dataset = np.load(data_path)

        self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
        self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()

        batch_size, c_len, w_len = self.context_char_idxs.size()
        ones = torch.ones((batch_size, 1), dtype=torch.int64)
        self.context_idxs = torch.cat((ones, self.context_idxs), dim=1)
        self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)

        ones = torch.ones((batch_size, 1, w_len), dtype=torch.int64)
        self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
        self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)

        self.y1s += 1
        self.y2s += 1

        # SQuAD 1.1: Ignore no-answer examples
        self.ids = torch.from_numpy(dataset['ids']).long()
        self.valid_idxs = [idx for idx in range(len(self.ids))
                           if use_v2 or self.y1s[idx].item() >= 0]

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]