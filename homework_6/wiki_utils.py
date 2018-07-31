import os
import torch
from torch.utils.data import DataLoader


class Alphabet(object):
    def __init__(self):
        self.symbol2idx = {}
        self.idx2symbol = []
        self._len = 0
        
    def add_symbol(self, s):
        if s not in self.symbol2idx:
            self.idx2symbol.append(s)
            self.symbol2idx[s] = self._len
            self._len += 1
    
    def __len__(self):
        return self._len


class Texts(object):
    def __init__(self, path):
        self.dictionary = Alphabet()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
      
        assert os.path.exists(path)
        # Add symbol to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                tokens += len(line)
                for s in line:
                    self.dictionary.add_symbol(s)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                for s in line:
                    ids[token] = self.dictionary.symbol2idx[s]
                    token += 1

        return ids
    

class TextLoader(object):
    def __init__(self, dataset, batch_size=128, sequence_length=30):
        self.data = dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self._batchify()
        
    def _batchify(self):
        # Work out how cleanly we can divide the dataset into batch_size parts.
        self.nbatch = self.data.size(0) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = self.data.narrow(0, 0, self.nbatch * self.batch_size)
        # Evenly divide the data across the batch_size batches.
        self.batch_data = data.view(self.batch_size, -1).t().contiguous()
    
    def _get_batch(self, i):
        seq_len = min(self.sequence_length, len(self.batch_data) - 1 - i)
        data = self.batch_data[i:i+seq_len]
        target = self.batch_data[i+1:i+1+seq_len].view(-1)
        return data, target
    
    def __iter__(self):
        for i in range(0, self.batch_data.size(0) - 1, self.sequence_length):
            data, targets = self._get_batch(i)
            yield data, targets
    
    def __len__(self):
        return self.batch_data.size(0)






class TextDataset(torch.utils.data.Dataset):
    
    raw_folder = 'raw'    
    training_file = 'train.txt'
    valid_file = 'valid.txt'
    test_file = 'test.txt'
    
    def __init__(self, root, dataset_type='train', alphabet=Alphabet()):
        self.root = os.path.expanduser(root)
        self.type = dataset_type  # training set, valid set or test set
        self.dictionary = alphabet
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.type_map = {
            'train': self.training_file,
            'valid': self.valid_file,
            'test': self.test_file
        }
        self.data = self.tokenize(os.path.join(self.root, self.type_map[self.type]))

    def tokenize(self, path):

        assert os.path.exists(path)
        # Add symbol to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                tokens += len(line)
                for s in line:
                    self.dictionary.add_symbol(s)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                for s in line:
                    ids[token] = self.dictionary.symbol2idx[s]
                    token += 1

        return ids

    def __len__(self):
        return len(str(self.data))

    def getData(self):
        return self.data

    def __getitem__(self, index):
       pass

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))


class TextDataLoader(torch.utils.data.DataLoader):

    __initialized = False

    def __init__(self, dataset, batch_size=128, sequence_length=32, sampler=None, batch_sampler=None,
                 num_workers=0, drop_last=False, collate_fn=torch.utils.data.dataloader.default_collate, pin_memory=False,
                 timeout=0, worker_init_fn=None):
        self.dataset = dataset
        self.data = dataset.getData()
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self._batchify()

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if batch_sampler is not None:
            if batch_size > 1  or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = None

        if self.num_workers < 0:
            raise ValueError('num_workers option cannot be negative; '
                             'use num_workers=0 to disable multiprocessing.')

        self.__initialized = True

    def _batchify(self):
        # Work out how cleanly we can divide the dataset into batch_size parts.
        self.nbatch = self.data.size(0) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = self.data.narrow(0, 0, self.nbatch * self.batch_size)
        # Evenly divide the data across the batch_size batches.
        self.batch_data = data.view(self.batch_size, -1).t().contiguous()

    def _get_batch(self, i):
        seq_len = min(self.sequence_length, len(self.batch_data) - 1 - i)
        data = self.batch_data[i:i + seq_len]
        target = self.batch_data[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    def __iter__(self):
        for i in range(0, self.batch_data.size(0) - 1, self.sequence_length):
            data, targets = self._get_batch(i)
           # targets = targets.view_as(data)
            yield data, targets

    def __len__(self):
        return len(self.batch_data)


def text_dataloader(path='./text', batch_size=128, sequence_length=32, valid=True, alphabet=Alphabet()):
    test_dataset = TextDataset(path, dataset_type='test', alphabet=alphabet)
    test_loader = TextDataLoader(test_dataset, batch_size=batch_size, sequence_length=sequence_length)

    train_dataset = TextDataset(path, dataset_type='train', alphabet=alphabet)
    if valid:
        valid_dataset = TextDataset(path, dataset_type='valid', alphabet=alphabet)
        train_loader = TextDataLoader(train_dataset, batch_size=batch_size, sequence_length=sequence_length)
        valid_loader = TextDataLoader(valid_dataset, batch_size=batch_size, sequence_length=sequence_length)
        return train_loader, valid_loader, test_loader
    else:
        train_loader = TextDataLoader(train_dataset, batch_size=batch_size, sequence_length=sequence_length)
        return train_loader, test_loader