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
import torchtext


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

def grad_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def train(model, iters, opt, criterion, optim):
    train_iter = iters['train']
    valid_iter = iters['valid']

    examples = train_iter.dataset.examples
    nsamples = len(examples)
    dist_is = np.array([1/nsamples for _ in range(nsamples)])
    bsz = train_iter.batch_size

    scores = np.zeros((nsamples, ))
    idx_score_filled = set()
    for epoch in range(opt.nepoch):
        for i in range(len(train_iter)):
            model.train()

            idx_sampled = np.random.choice(nsamples, bsz, p=dist_is)
            idx_sampled = sorted(idx_sampled, key=lambda i: -len(examples[i].txt))
            examples_sampled = [examples[i] for i in idx_sampled]
            batch = torchtext.data.Batch(examples_sampled,
                                         train_iter.dataset,
                                         train_iter.device)

            txt, lbl = batch.txt, batch.lbl
            probs = model(txt)
            loss_batch = criterion(probs, lbl.squeeze(0))
            for idx, loss in zip(idx_sampled, loss_batch):
                model.zero_grad()
                loss = loss/len(idx_sampled)
                loss.backward(retain_graph=True)
                clip_grad_norm_(model.parameters(), 5)
                scores[idx] = grad_norm(model.parameters())
                idx_score_filled.add(idx)
                optim.step()

            utils.progress_bar(i / len(train_iter),
                               (loss_batch.sum()/len(idx_sampled)).item(),
                               epoch)

        # update the distribution
        val_smooth = min([scores[i] for i in idx_score_filled])
        for i in range(len(dist_is)):
            if i not in idx_score_filled:
                dist_is[i] = scores[i]
            else:
                dist_is[i] = val_smooth
        dist_is/=dist_is.sum()

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



