import nets
from macros import *
import argparse
import opts
import torch
import preproc
from torch import optim
import utils
from torch.nn.utils import clip_grad_norm_
from torch import nn
import crash_on_ipy
import sklearn
import numpy as np
import json
import os
from sklearn.metrics import f1_score, \
    precision_score, \
    recall_score, \
    accuracy_score
import training

if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(description='main.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)

    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    d0_train = os.path.join(CHEN, '0.train')
    d0_valid = os.path.join(CHEN, '0.valid')

    TXT, train_iter, valid_iter = \
        preproc.build_iters(ftrain=d0_train,
                            fvalid=d0_valid,
                            emb_pretrain=opt.pretrain,
                            skip_header=False,
                            bsz=opt.bsz,
                            min_freq=opt.min_freq,
                            device=opt.gpu)

    model = None

    if opt.net == 'bigru':
        model = nets.BiRNN(voc_size=len(TXT.vocab.itos),
                             edim=opt.edim,
                             hdim=opt.hdim,
                             dropout=opt.dropout,
                             padding_idx=TXT.vocab.stoi[PAD]).to(device)
    if opt.net == 'max_pooling':
        model = nets.MaxPooling(voc_size=len(TXT.vocab.itos),
                             edim=opt.edim,
                             hdim=opt.hdim,
                             dropout=opt.dropout,
                             padding_idx=TXT.vocab.stoi[PAD]).to(device)
    if opt.net == 'avg_pooling':
        model = nets.AvgPooling(voc_size=len(TXT.vocab.itos),
                                edim=opt.edim,
                                hdim=opt.hdim,
                                dropout=opt.dropout,
                                padding_idx=TXT.vocab.stoi[PAD]).to(device)
    if opt.net == 'rnn_atten':
        model = nets.RNNAtteion(voc_size=len(TXT.vocab.itos),
                                edim=opt.edim,
                                hdim=opt.hdim,
                                dropout=opt.dropout,
                                padding_idx=TXT.vocab.stoi[PAD]).to(device)
    if opt.net == 'rnn_atten_lm':
        model = nets.RNNAtteionLM(voc_size=len(TXT.vocab.itos),
                                edim=opt.edim,
                                hdim=opt.hdim,
                                dropout=opt.dropout,
                                padding_idx=TXT.vocab.stoi[PAD]).to(device)

    utils.init_model(model)
    if opt.pretrain:
        model.embedding.weight.data.copy_(TXT.vocab.vectors)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr, weight_decay=opt.wdecay)

    weights = utils.balance_bias(train_iter)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(device))
    criterion_lm = nn.CrossEntropyLoss(ignore_index=TXT.vocab.stoi[PAD])

    folder_pwd = os.path.join(DATA, CHEN)
    info = json.loads(open(os.path.join(folder_pwd, INFO), "rt").read())

    training.train(model,
                   {'train':train_iter, 'valid':valid_iter},
                    opt,
                    0,
                   {'senti':criterion, 'lm':criterion_lm},
                    optimizer)



