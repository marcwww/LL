import torch
from sklearn.metrics import f1_score, \
    precision_score, \
    recall_score, \
    accuracy_score
from torch.nn.utils import clip_grad_norm_
import utils
import torchtext
import os
import sys
from macros import *
from torch import nn

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

            percent = i/len(train_iter)

            utils.progress_bar(percent, loss.item(), epoch)

            if (i+1) % int(1/4 * len(train_iter)) == 0 :
                # print('\r')
                accurracy, precision, recall, f1 = \
                    valid(model, valid_iter)
                print('{\'Epoch\':%d, \'Domain\':%d, \'Format\':\'a/p/r/f\', \'Metrics\':[%.4f, %.4f, %.4f, %.4f]}' %
                      (epoch, domain, accurracy, precision, recall, f1))


        if (epoch + 1) % opt.save_per == 0:
            basename = "up-to-domain-{}-epoch-{}".format(domain, epoch)
            model_fname = basename + ".model"
            torch.save(model.state_dict(), model_fname)

def train_domain(model, iters, opt, domain, criterion, optim):

    flog = opt.name + '.log'
    with open(os.path.join(RES, flog), 'a+') as print_to:

        train_iter = iters['train']
        valid_iters = iters['valids'][:domain+1]

        print('\r')
        print('{\'domain\':%d}' % domain)
        print('{\'domain\':%d}' % domain, file=print_to)
        # print('--' * 10 + ('domain %d' % domain) + '--' * 10)
        # print('--' * 10 + ('domain %d' % domain) + '--' * 10, file=print_to)

        best_model = ''
        best_f1 = 0
        best_metrics = {}
        best_epoch = 0
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

                if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                    # valid
                    accurracy, precision, recall, f1 = \
                        valid(model, valid_iters[domain])
                    if f1 > best_f1:
                        print('\r')
                        print(
                            '{\'Epoch\':%d, \'Domain\':%d, \'Format\':\'a/p/r/f\', \'Metrics\':[%4f, %4f, %4f, %4f]}' %
                            (epoch, domain, accurracy, precision, recall, f1))

                        # save model
                        basename = "up-to-domain-{}-epoch-{}".format(domain, epoch)
                        model_fname = basename + ".model"
                        torch.save(model.state_dict(), model_fname)

                        best_f1 = f1
                        best_model = model_fname
                        best_metrics[domain] = (accurracy, precision, recall, f1)
                        best_epoch = epoch

                        # valid the rest
                        for d, valid_iter in enumerate(valid_iters):
                            if d == domain:
                                continue

                            accurracy, precision, recall, f1 =\
                                valid(model, valid_iter)

                            print('{\'Epoch\':%d, \'Domain\':%d, \'Format\':\'a/p/r/f\', \'Metrics\':[%4f, %4f, %4f, %4f]}' %
                                  (epoch, d, accurracy, precision, recall, f1))

                            best_metrics[d] = (accurracy, precision, recall, f1)

                    print_to.flush()

        # logging the best performance on the current domain
        for d in range(domain+1):
            accurracy, precision, recall, f1 = best_metrics[d]
            print('{\'Epoch\':%d, \'Domain\':%d, \'Format\':\'a/p/r/f\', \'Metrics\':[%4f, %4f, %4f, %4f]}' %
                  (best_epoch, d, accurracy, precision, recall, f1), file=print_to)

    location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
    model_dict = torch.load(best_model, map_location=location)
    model.load_state_dict(model_dict)

def train_ll(model, uiters, info, opt, optim):

    flog = opt.name + '.log'
    with open(os.path.join(RES, flog), 'w') as print_to:
        pass

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

        location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
        device = torch.device(location)

        weights = utils.balance_bias(train_iters[domain])
        print(weights)
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(device))
        train_domain(model, {'train': train_iters[domain],
                             'valids': valid_iters},
                     opt, domain, criterion, optim)

