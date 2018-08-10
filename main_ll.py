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
import training_text, training_cv
from torchvision import datasets, transforms

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
    train_loader = None
    valid_loader = None

    if opt.dataset == CHEN:
        TXT, utrain_iter, uvalid_iter = \
            preproc.build_iters_CHEN(ftrain=opt.ftrain,
                                fvalid=opt.fvalid,
                                emb_pretrain=opt.pretrain,
                                skip_header=False,
                                bsz=opt.bsz,
                                min_freq=opt.min_freq,
                                device=opt.gpu)
    if opt.dataset == MAN:
        TXT, utrain_iter, uvalid_iter = \
            preproc.build_iters_MAN(ftrain=opt.ftrain,
                                fvalid=opt.fvalid,
                                emb_pretrain=opt.pretrain,
                                skip_header=False,
                                bsz=opt.bsz,
                                min_freq=opt.min_freq,
                                device=opt.gpu)

    if opt.dataset == MNIST:
        # kwargs = {'num_workers': 1, 'pin_memory': True} \
        #     if torch.cuda.is_available() and opt.gpu != -1 else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=opt.bsz, shuffle=False,)
            # **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=opt.bsz, shuffle=False,)
            # **kwargs)

    model = None
    nclasses = opt.nclasses

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
    if opt.net == 'mlp':
        model = nets.MLP(opt.idim, nclasses).to(device)

    if opt.net == 'rammlp':
        model = nets.RAMMLP(idim=opt.idim,nclasses=nclasses,
                            capacity=opt.capacity,
                            criterion = nn.CrossEntropyLoss(),
                            add_per=opt.add_per).to(device)

    if opt.net == 'mbpamlp':
        model = nets.MbPAMLP(idim=opt.idim,nclasses=nclasses,
                            capacity=opt.capacity,
                            criterion = nn.CrossEntropyLoss(),
                            add_per=opt.add_per,
                            device=device).to(device)

    if opt.net == 'mbpamlp2layers':
        model = nets.MbPAMLP2Layers(idim=opt.idim,nclasses=nclasses,
                            capacity=opt.capacity,
                            criterion = nn.CrossEntropyLoss(),
                            add_per=opt.add_per,
                            device=device).to(device)

    if opt.net == 'gnimlp':
        model = nets.GNIMLP(idim=opt.idim,
                            nclasses=nclasses,
                            capacity=opt.capacity,
                            criterion=nn.CrossEntropyLoss(),
                            add_per=opt.add_per,
                            retain_ratio=opt.retain_ratio,
                            device=device).to(device)

    utils.init_seed(opt.seed)
    utils.init_model(model)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr)

    criterion = nn.CrossEntropyLoss()

    if opt.dataset in TEXT_DATASETS:
        folder_pwd = os.path.join(DATA, CHEN)
        info = json.loads(open(os.path.join(folder_pwd, INFO), "rt").read())
        training_text.train_ll(model,
                          {'train':utrain_iter, 'valid':uvalid_iter},
                          info,
                          opt,
                          optimizer)

    if opt.dataset in CV_DATASETS:

        training_cv.train_ll_mnist(model,
                                   {'train':train_loader,
                                     'valid':valid_loader},
                                   opt,
                                   optimizer)




