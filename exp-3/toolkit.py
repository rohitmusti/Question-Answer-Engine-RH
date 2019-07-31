import ujson as json
import numpy as np
import torch 
import torch.utils.data as data
import logging
from tqdm import tqdm
import os

class qcd(data.Dataset):
    def __init__(self, data_path):
        """
        idea: sentiment as a feature
        """
        dataset = np.load(data_path)
        self.qw_idxs = torch.from_numpy(dataset['qw_idxs']).long()
        self.ids = torch.from_numpy(dataset['ids']).long()
        self.topic_ids = torch.from_numpy(dataset['topic_ids']).long()
        # NOTE: every idx is valid so no need for the valid idx array they used
    def __getitem__(self, idx):
        example = {self.qw_idxs[idx],
                   self.ids[idx],
                   self.topic_ids[idx]}
        return example
    
    def __len__(self):
        return len(self.ids)

def collate_fn(examples):
    """
    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded
    qw_idxs, ids, topic_ids = zip(*examples)
    qw_idxs = merge_1d(qw_idxs)
    ids = merge_0d(ids)
    topic_ids = merge_0d(topic_ids)

    return (qw_idxs, ids, topic_ids)



def save(filename, obj):
    """
    just saves the file, nothing fancy
    author: @wzhouad
    """
    with open(filename, "w") as fh:
        json.dump(obj, fh)

def quick_clean(raw_str):
    """
    args:
        - raw_str: a string to be quickly cleaned

    return
        - the original string w/ all quotes replaced as double quotes
    """
    return raw_str.replace("''", '" ').replace("``", '" ').strip()

def get_logger(log_dir, name):
    """
    from @chrischute

    Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
