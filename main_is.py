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
import torchtext
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

    bsz = train_iter.batch_size
    bsz_pre = 64
    examples = train_iter.dataset.examples
    nsamples = len(examples)
    device = train_iter.device

    for epoch in range(opt.nepoch):
        for batch_idx in range(len(train_iter)):
            model.train()
            model.zero_grad()

            idx_pre = np.random.choice(nsamples, bsz_pre)
            idx_pre = sorted(idx_pre, key=lambda i: -len(examples[i].txt))
            dist_pre = []
            idx_dict = {}
            for i in idx_pre:
                idx_dict[i] = len(idx_dict)

            examples_pre = [train_iter.dataset.examples[i] for i in idx_pre]

            fields_pre = train_iter.dataset.fields
            data_pre = torchtext.data.Dataset(examples_pre, fields_pre)
            batches_pre = torchtext.data.batch(data_pre, bsz)

            losses_pre = []
            for minibatch in batches_pre:
                batch = torchtext.data.Batch(minibatch, data_pre, device)
                txt, lbl = batch.txt, batch.lbl
                probs = model(txt)
                loss = criterion(probs, lbl.squeeze(0))
                losses_pre.extend(loss)
                dist_pre.extend(loss.data.numpy())

            dist_pre = np.array(dist_pre)
            # normalize
            dist_pre = dist_pre/dist_pre.sum()
            idx = np.random.choice(idx_pre, bsz, p=dist_pre)
            idx = sorted(idx, key=lambda i: -len(examples[i].txt))
            loss = 0
            normalizer = 0
            for i in idx:
                loss += losses_pre[idx_dict[i]]
                # w = 1/(dist_pre[idx_dict[i]] * bsz_pre)
                # normalizer += w
                # loss += losses_pre[idx_dict[i]] * w
            loss/=len(idx)
            # loss/=normalizer
            loss.backward()

            clip_grad_norm_(model.parameters(), 5)
            optim.step()
            utils.progress_bar(batch_idx / len(train_iter), loss.item(), epoch)

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

    criterion = nn.CrossEntropyLoss(reduce=False)
    train(model, {'train': train_iter,
                  'valid': valid_iter},
          opt,
          criterion,
          optimizer)



