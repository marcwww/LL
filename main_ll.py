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

    TXT = None
    utrain_iter = None
    uvalid_iter = None

    if opt.dataset == 'CHEN':
        TXT, utrain_iter, uvalid_iter = \
            preproc.build_iters_CHEN(ftrain=opt.ftrain,
                                fvalid=opt.fvalid,
                                emb_pretrain=opt.pretrain,
                                skip_header=False,
                                bsz=opt.bsz,
                                min_freq=opt.min_freq,
                                device=opt.gpu)
    if opt.dataset == 'MAN':
        TXT, utrain_iter, uvalid_iter = \
            preproc.build_iters_MAN(ftrain=opt.ftrain,
                                fvalid=opt.fvalid,
                                emb_pretrain=opt.pretrain,
                                skip_header=False,
                                bsz=opt.bsz,
                                min_freq=opt.min_freq,
                                device=opt.gpu)

    model = None
    nclasses = len(utrain_iter.dataset.fields['lbl'].vocab.itos)

    if opt.net == 'bigru':
        model = nets.BiRNN(voc_size=len(TXT.vocab.itos),
                           edim=opt.edim,
                           hdim=opt.hdim,
                           dropout=opt.dropout,
                           padding_idx=TXT.vocab.stoi[PAD],
                           nclasses=nclasses).to(device)
    if opt.net == 'max_pooling':
        model = nets.MaxPooling(voc_size=len(TXT.vocab.itos),
                           edim=opt.edim,
                           hdim=opt.hdim,
                           dropout=opt.dropout,
                           padding_idx=TXT.vocab.stoi[PAD],
                           nclasses=nclasses).to(device)
    if opt.net == 'avg_pooling':
        model = nets.AvgPooling(voc_size=len(TXT.vocab.itos),
                           edim=opt.edim,
                           hdim=opt.hdim,
                           dropout=opt.dropout,
                           padding_idx=TXT.vocab.stoi[PAD],
                           nclasses=nclasses).to(device)
    if opt.net == 'rnn_atten':
        model = nets.RNNAtteion(voc_size=len(TXT.vocab.itos),
                           edim=opt.edim,
                           hdim=opt.hdim,
                           dropout=opt.dropout,
                           padding_idx=TXT.vocab.stoi[PAD],
                           nclasses=nclasses).to(device)
    if opt.net == 'rnn_atten_lm':
        model = nets.RNNAtteionLM(voc_size=len(TXT.vocab.itos),
                           edim=opt.edim,
                           hdim=opt.hdim,
                           dropout=opt.dropout,
                           padding_idx=TXT.vocab.stoi[PAD],
                           nclasses=nclasses).to(device)

    utils.init_model(model)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr)

    criterion = nn.CrossEntropyLoss()

    folder_pwd = os.path.join(DATA, CHEN)
    info = json.loads(open(os.path.join(folder_pwd, INFO), "rt").read())

    utils.init_seed(opt.seed)
    training.train_ll(model,
                      {'train':utrain_iter, 'valid':uvalid_iter},
                      info,
                      opt,
                      optimizer)


