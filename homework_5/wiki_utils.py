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
        """Tokenizes a text file."""
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
    
    def __init__(self, root, train=True, sequence_length=32):
        self.root = os.path.expanduser(root)
        self.sequence_length = sequence_length
        self.train = train  # training set or test set
        self.dictionary = Alphabet()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            self.train_data = self.tokenize(os.path.join(self.root, self.training_file))
            self.valid_data = self.tokenize(os.path.join(self.root, self.valid_file))
        else:
            self.test_data = self.tokenize(os.path.join(self.root, self.test_file))

    def tokenize(self, path):
        """Tokenizes a text file."""
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
        if self.train:
            return len(str(self.train_data))
        else:
            return len(str(self.test_data))

    def get_seq(self, dataset):
        for i in range(0, dataset.size(0) - 1, self.sequence_length):
            seq_len = min(self.sequence_length, len(dataset) - 1 - i)
            data = dataset[i:i + seq_len]
            target = dataset[i + 1:i + 1 + seq_len].view(-1)
            yield data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        texts = []
        targets = []
        for batch, (data_seq, target_seq) in enumerate(self.get_seq(self.train_data)):
            texts.append(data_seq)
            targets.append(target_seq)


        #if self.train:
        #    text, target = self.train_data[index], self.train_data[index]
        #else:
        #    text, target = self.test_data[index], self.test_data[index]

        return texts, targets

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))



def text_dataloader(path='./text', batch_size=128, valid=0):
    test_data = TextDataset(path, train=False)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    train_data = TextDataset(path, train=True)
    if valid > 0:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = num_train - valid

        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

        return train_loader, valid_loader, test_loader
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size)
        return train_loader, test_loader