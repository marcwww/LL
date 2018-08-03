import torchtext
import os
from macros import *
import random
import json
import crash_on_ipy

def build_iters(ftrain, fvalid, skip_header, bsz, device, min_freq):

    TXT = torchtext.data.Field(sequential=True,
                               pad_token=PAD,
                               unk_token=UNK)

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
    TXT.build_vocab(train, min_freq=min_freq)
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



if __name__ == '__main__':
    split(CHEN, {'train':0.8,
                 'valid':0.1,
                 'test':0.1})

    unify(CHEN, 'train')
    unify(CHEN, 'valid')
    unify(CHEN, 'test')











