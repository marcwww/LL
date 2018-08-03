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
from sklearn.metrics import f1_score, \
    precision_score, \
    recall_score, \
    accuracy_score

def valid(model, valid_iter):
    model.eval()
    pred_lst = []
    true_lst = []

    with torch.no_grad():
        for i, sample in enumerate(valid_iter):
            txt, lbl = sample.txt, sample.lbl
            lbl = lbl.squeeze(0)
            # probs: (bsz, 3)
            probs = model(txt)
            pred = probs.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            pred_lst.extend(pred)
            true_lst.extend(lbl)

    accurracy = accuracy_score(true_lst, pred_lst)
    precision = precision_score(true_lst, pred_lst, average='macro')
    recall = recall_score(true_lst, pred_lst, average='macro')
    f1 = f1_score(true_lst, pred_lst, average='macro')

    return accurracy, precision, recall, f1

def train(model, iters, opt, criterion, optim):
    train_iter = iters['train']
    valid_iter = iters['valid']

    for epoch in range(opt.nepoch):
        for i, sample in enumerate(train_iter):
            model.train()
            txt, lbl = sample.txt, sample.lbl

            model.zero_grad()
            # probs: (bsz, 3)
            probs = model(txt)
            
            loss = criterion(probs, lbl.squeeze(0))

            loss.backward()
            clip_grad_norm_(model.parameters(), 5)
            optim.step()

            utils.progress_bar(i/len(train_iter), loss.item(), epoch)
        print('\n')

        print(valid(model, valid_iter))

        if (epoch + 1) % opt.save_per == 0:
            basename = "{}-epoch-{}".format(opt.name, epoch)
            model_fname = basename + ".model"
            torch.save(model.state_dict(), model_fname)

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

    TXT, train_iter, valid_iter = \
        preproc.build_iters(ftrain=opt.ftrain,
                            fvalid=opt.fvalid,
                            bsz=opt.bsz,
                            min_freq=opt.min_freq,
                            device=opt.gpu)

    model = nets.BaseRNN(voc_size=len(TXT.vocab.itos),
                         edim=opt.edim,
                         hdim=opt.hdim,
                         padding_idx=TXT.vocab.stoi[PAD]).to(device)

    utils.init_model(model)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr)

    criterion = nn.CrossEntropyLoss()
    train(model, {'train': train_iter,
                  'valid': valid_iter},
          opt,
          criterion,
          optimizer)



