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

    model = nets.BaseRNN(voc_size=len(TXT.vocab.itos),
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

    folder_pwd = os.path.join(DATA, CHEN)
    info = json.loads(open(os.path.join(folder_pwd, INFO), "rt").read())

    training.train(model,
                   {'train':train_iter, 'valid':valid_iter},
                    opt,
                    0,
                    criterion,
                    optimizer)



