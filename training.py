import torch
from sklearn.metrics import f1_score, \
    precision_score, \
    recall_score, \
    accuracy_score
from torch.nn.utils import clip_grad_norm_
import utils
import torchtext
import os
from macros import *

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

def train(model, iters, opt, domain, criterion, optim):
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

            utils.progress_bar(i / len(train_iter), loss.item(), epoch)
        print('\n')

        print(valid(model, valid_iter))

        if (epoch + 1) % opt.save_per == 0:
            basename = "up-to-domain-{}-epoch-{}".format(domain, epoch)
            model_fname = basename + ".model"
            torch.save(model.state_dict(), model_fname)

def train_domain(model, iters, opt, domain, criterion, optim):

    flog = opt.name + '.log'
    with open(os.path.join(RES, flog), 'a+') as print_to:

        train_iter = iters['train']
        valid_iters = iters['valids'][:domain+1]

        print('--'*10+('domain %d' % domain)+'--'*10, file=print_to)
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

                utils.progress_bar(i / len(train_iter), loss.item(), epoch)

            print('\n', file=print_to)
            for d, valid_iter in enumerate(valid_iters):
                accurracy, precision, recall, f1 =\
                    valid(model, valid_iter)
                print('Domain(%d/%d): a/p/r/f [%4f, %4f, %4f, %4f]' %
                      (d, domain, accurracy, precision, recall, f1), file=print_to)

            if (epoch + 1) % opt.save_per == 0:
                basename = "up-to-domain-{}-epoch-{}".format(domain, epoch)
                model_fname = basename + ".model"
                torch.save(model.state_dict(), model_fname)

def train_ll(model, uiters, info, opt, criterion, optim):

    utrain_iter = uiters['train']
    uvalid_iter = uiters['valid']
    train_ranges = info['train_ranges']
    valid_ranges = info['valid_ranges']
    fields = utrain_iter.dataset.fields
    bsz = utrain_iter.batch_size
    device = utrain_iter.device

    domains = range(len(train_ranges))

    examples_utrain = utrain_iter.dataset.examples
    examples_uvalid = uvalid_iter.dataset.examples
    train_iters = []
    valid_iters = []
    for domain in domains:
        begin_train, end_train = train_ranges[domain]
        begin_valid, end_valid = valid_ranges[domain]

        examples_train = examples_utrain[begin_train: end_train]
        examples_valid = examples_uvalid[begin_valid: end_valid]

        train = torchtext.data.Dataset(examples_train, fields)
        valid = torchtext.data.Dataset(examples_valid, fields)

        train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                sort=False, repeat=False,
                                sort_key=lambda x: len(x.txt),
                                sort_within_batch=True,
                                shuffle=False,
                                device=device)
        valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                             sort=False, repeat=False,
                                             sort_key=lambda x: len(x.txt),
                                             sort_within_batch=True,
                                             shuffle=False,
                                             device=device)

        train_iters.append(train_iter)
        valid_iters.append(valid_iter)

    for domain in domains:
        train_domain(model, {'train': train_iters[domain],
                             'valids': valid_iters},
                     opt, domain, criterion, optim)

