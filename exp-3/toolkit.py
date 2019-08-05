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

#        dataset = torch.load("./data/torch-test-train")
#        temp_qw_idxs = dataset['qw_idxs'].long()
#        temp_ids = dataset['ids'].long()
#        temp_topic_ids = dataset['topic_ids'].long()
#
#        print(f"QW_IDXS match: {torch.equal(self.qw_idxs, temp_qw_idxs)}")
#        print(f"IDS match: {torch.equal(self.ids, temp_ids)}")
#        print(f"TOPIC_IDS match: {torch.equal(self.topic_ids, temp_topic_ids)}")

        # NOTE: every idx is valid so no need for the valid idx array they used
    def __getitem__(self, idx):
        temp_qw_idx = self.qw_idxs[idx]
        temp_id = self.ids[idx]
        temp_tid = self.topic_ids[idx]
        # print(temp_qw_idx)
        # print(temp_id)
        # print(temp_tid)
        if temp_tid >= 442 or temp_tid < 0:
            raise ValueError("The value for the topic index {self.topic_ids[idx]} is outside the allowed range")
        # using this dictionary method results in non-deterministic computing
        # example = {temp_qw_idx,
        #            temp_id,
        #            temp_tid}
        # print(example)
        return temp_qw_idx, temp_id, temp_tid
    
    def __len__(self):
        return len(self.ids)

def collate_fn(examples):
    """
    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0, pad_length=31):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), pad_length, dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded, torch.tensor(lengths)

    qw_idxs, ids, topic_ids = zip(*examples)
#    print(f"qw_idxs: {qw_idxs}")
#    print(f"ids: {ids}")
#    print(f"topic_ids: {topic_ids}")
    qw_idxs, lengths = merge_1d(qw_idxs)
    ids = merge_0d(ids)
    topic_ids = merge_0d(topic_ids)

    return (qw_idxs, ids, topic_ids, lengths)

class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')

class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


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

class CheckpointSaver:
    def __init__(self, save_dir, max_checkpoints, metric_name, log=None):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.checkpoints = []
        self.log = log
        self.best_val = None
        self._print(f"Saver will minimize {metric_name}...")
    
    def _is_best(self, metric_val):
        if metric_val is None:
            return False
        elif self.best_val is None :
            self.best_val = metric_val
            self.log.info(f"The best value has been updated to {metric_val}")
            return True
        elif self.best_val <= metric_val:
            return False
        elif metric_val < self.best_val:
            self.best_val = metric_val
            self.log.info(f"The best value has been updated to {metric_val}")
            return True
        else:
            return False
    
    def _print(self, message):
        self.log.info(message)

    def save(self, model, step, epoch, metric_val, device):
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step,
            'epoch': epoch,
            'BCELoss Val': metric_val
        }

        if self._is_best(metric_val):
            checkpoint_path = os.path.join(self.save_dir, f'best.pth.tar')
            torch.save(ckpt_dict, checkpoint_path)
            self._print(f"Saved best checkpoint: {checkpoint_path}")

        checkpoint_path = os.path.join(self.save_dir, f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f"Saved {step} checkpoint: {checkpoint_path}")

        self.checkpoints.append((checkpoint_path, metric_val))

        if len(self.checkpoints) > self.max_checkpoints:
            lowest = 100000000
            worst = 0
            for i, tup in enumerate(self.checkpoints):
                if lowest > tup[1]:
                    lowest = tup[1]
                    worst = i
            self.checkpoints.remove(self.checkpoints[worst])
            self.log.info(f'Removed worst checkpoint with loss of {tup[1]}')


        

        
