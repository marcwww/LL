import torchtext
import os
from macros import *
import random
import json
import torch
from torchvision import datasets, transforms
import crash_on_ipy
import codecs
import numpy as np

def build_iters_CHEN(ftrain, fvalid, emb_pretrain, skip_header, bsz, device, min_freq):

    TXT = torchtext.data.Field(sequential=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=EOS)

    LBL = torchtext.data.Field(sequential=True, use_vocab=True, eos_token=None, pad_token=None, unk_token=None)
    DOM = torchtext.data.Field(sequential=True, use_vocab=True)
    RAT = torchtext.data.Field(sequential=False, use_vocab=False)

    train = torchtext.data.TabularDataset(path=os.path.join(DATA, ftrain),
                                          format='tsv',
                                          fields=[('dom', DOM),
                                                  ('lbl', LBL),
                                                  ('rat', RAT),
                                                  ('txt', TXT)],
                                          skip_header=skip_header)

    LBL.build_vocab(train)
    DOM.build_vocab(train)
    TXT.build_vocab(train, min_freq=min_freq, vectors=emb_pretrain)
    valid = torchtext.data.TabularDataset(path=os.path.join(DATA, fvalid),
                                          format='tsv',
                                          fields=[('dom', DOM),
                                                  ('lbl', LBL),
                                                  ('rat', RAT),
                                                  ('txt', TXT)],
                                          skip_header=skip_header)

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x:len(x.txt),
                                         sort_within_batch=True,
                                         shuffle=False,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.txt),
                                         sort_within_batch=True,
                                         shuffle=False,
                                         device=device)

    return TXT, train_iter, valid_iter

def build_iters_MAN(ftrain, fvalid, emb_pretrain, skip_header, bsz, device, min_freq):

    TXT = torchtext.data.Field(sequential=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=EOS)

    LBL = torchtext.data.Field(sequential=True, use_vocab=True, eos_token=None, pad_token=None, unk_token=None)

    train = torchtext.data.TabularDataset(path=os.path.join(DATA, ftrain),
                                          format='tsv',
                                          fields=[('lbl', LBL),
                                                  ('txt', TXT)],
                                          skip_header=skip_header)

    LBL.build_vocab(train)
    TXT.build_vocab(train, min_freq=min_freq, vectors=emb_pretrain)
    valid = torchtext.data.TabularDataset(path=os.path.join(DATA, fvalid),
                                          format='tsv',
                                          fields=[('lbl', LBL),
                                                  ('txt', TXT)],
                                          skip_header=skip_header)

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x:len(x.txt),
                                         sort_within_batch=True,
                                         shuffle=False,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.txt),
                                         sort_within_batch=True,
                                         shuffle=False,
                                         device=device)

    return TXT, train_iter, valid_iter

def split(folder, ratio):
    root = os.path.join(DATA, folder)
    f2i = {}
    i2f = []
    for fname in os.listdir(root):
        f2i[fname] = len(f2i)
        i2f.append(fname)
        with open(os.path.join(root, fname), 'r') as f:
            lines = f.readlines()
            # pop the header
            lines.pop(0)
            lines = random.sample(lines, k=len(lines))

            assert ratio['train'] + ratio['valid'] + ratio['test'] == 1, \
                'the split ratio is illegal'

            pnt0, pnt1 = int(ratio['train']*len(lines)), \
                         int((ratio['train']+ratio['valid'])*len(lines))

            train, valid, test = lines[:pnt0], \
                                 lines[pnt0:pnt1], \
                                 lines[pnt1:]
            with open(os.path.join(root, str(f2i[fname])+'.train'), 'w') as ftrain, \
                    open(os.path.join(root, str(f2i[fname]) + '.valid'), 'w') as fvalid, \
                    open(os.path.join(root, str(f2i[fname]) + '.test'), 'w') as ftest:

                ftrain.writelines(train)
                fvalid.writelines(valid)
                ftest.writelines(test)

    info = \
        {'f2i':f2i,
        'i2f':i2f,
        'ratio':ratio}

    with open(os.path.join(root, 'info.json'), 'wt') as f:
        f.write(json.dumps(info))

def unify(folder, category='train'):

    folder_pwd = os.path.join(DATA, folder)
    info = json.loads(open(os.path.join(folder_pwd, INFO), "rt").read())
    f2i = info['f2i']

    lines = []
    info[category + '_ranges'] = []
    for fname, fidx in f2i.items():
        with open(os.path.join(folder_pwd, str(fidx) + '.' + category), 'r') as f:
            begin = len(lines)
            lines_train = f.readlines()
            lines.extend(lines_train)
            end = len(lines)
            info[category + '_ranges'].append((begin, end))

    with open(os.path.join(folder_pwd, 'unify' + '.' +category), 'w') as f:
        f.writelines(lines)

    with open(os.path.join(folder_pwd, 'info.json'), 'wt') as f:
        f.write(json.dumps(info))

def index(folder, ratio, encoding='ISO-8859-2'):
    root = os.path.join(DATA, folder)
    f2i = {}
    i2f = []
    for fname in os.listdir(root):
        fname_components = fname.split('.')
        if len(fname_components) != 3:
            continue

        dname, suffix = fname_components[0], \
                        fname_components[-1]
        if dname not in f2i.keys():
            f2i[dname] = len(f2i)
            i2f.append(dname)

        with open(os.path.join(root, fname), 'r', encoding=encoding) as f:
            lines = f.readlines()

        with open(os.path.join(root, str(f2i[dname])+'.'+str(suffix)), 'w') as f:
            f.writelines(lines)

    info = \
        {'f2i': f2i,
         'i2f': i2f,
         'ratio': ratio}

    with open(os.path.join(root, 'info.json'), 'wt') as f:
        f.write(json.dumps(info))

class index_iter(object):

    def __init__(self, data, bsz, shuffle=True):
        self.data = data
        self.bsz = bsz
        self.shuffle = shuffle
        self.len = len(data)
        self.batch_len = int(self.len/bsz)
        self.batch_idx = 0
        self.idx_seq = self._gen_idx_seq()

    def __iter__(self):
        return self

    def _gen_idx_seq(self):
         return np.random.choice(self.len, self.len) \
            if self.shuffle \
            else range(self.len)

    def _restart(self):
        self.batch_idx = 0
        self.idx_seq = self._gen_idx_seq()

    def __len__(self):
        return self.batch_len

    def __next__(self):
        if self.batch_idx < self.batch_len:
            start = self.batch_idx * self.bsz
            batch_x = []
            batch_y = []
            idices = []
            for offset in range(self.bsz):
                idx = start + offset
                if idx >= self.len:
                    self._restart()
                    raise StopIteration()

                batch_x.append(self.data[self.idx_seq[idx]][0])
                batch_y.append(self.data[self.idx_seq[idx]][1].unsqueeze(0))
                idices.append(self.idx_seq[idx])

            self.batch_idx += 1

            return idices, \
                   torch.cat(batch_x, dim=0), \
                   torch.cat(batch_y, dim=0)

        self._restart()
        raise StopIteration()

def build_iters_iMNIST(mnist_folder, bsz, device):

    train = datasets.MNIST(mnist_folder, train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    test = datasets.MNIST(mnist_folder, train=False,
                   transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    train.train_data = train.train_data.to(device)
    test.test_data = test.test_data.to(device)

    return index_iter(train, bsz), \
           index_iter(test, bsz)

if __name__ == '__main__':
    # for CHEN:
    # split(CHEN, {'train':0.8,
    #              'valid':0.1,
    #              'test':0.1})
    #
    # unify(CHEN, 'train')
    # unify(CHEN, 'valid')
    # unify(CHEN, 'test')

    # for MAN:
    # index(MAN, {'train':0.8,'test':0.2,'valid':0})
    # unify(MAN, 'train')
    # unify(MAN, 'test')

    device = torch.device('cpu')
    train_iter, test_iter = build_iters_iMNIST(os.path.join(DATA, MNIST), 32, device)

    for i, (indices, batch_x, batch_y) in enumerate(train_iter):
        print(i)

    for i, (indices, batch_x, batch_y) in enumerate(train_iter):
        print(i)









