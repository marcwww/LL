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
import numpy as np

def valid_mnist(model, valid_loader, task_permutation, device):
    model.eval()
    pred_lst = []
    true_lst = []

    with torch.no_grad():
        for i, (input, lbl) in enumerate(valid_loader):
            input = input.view(-1, MNIST_DIM)
            input = input[:, task_permutation].to(device)
            lbl = lbl.squeeze(0)
            # probs: (bsz, 3)

            out = model(input)

            pred = out.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            pred_lst.extend(pred)
            true_lst.extend(lbl)

    accurracy = accuracy_score(true_lst, pred_lst)
    precision = precision_score(true_lst, pred_lst, average='macro')
    recall = recall_score(true_lst, pred_lst, average='macro')
    f1 = f1_score(true_lst, pred_lst, average='macro')

    return accurracy, precision, recall, f1

def train_domain_mnist(model, dataloaders, opt, domain,
                       task_permutations, criterion, optim, device):

    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']

    flog = opt.name + '.log'
    with open(os.path.join(RES, flog), 'a+') as print_to:

        print('\r')
        print('{\'domain\':%d}' % domain)
        print('{\'domain\':%d}' % domain, file=print_to)
        # print('--' * 10 + ('domain %d' % domain) + '--' * 10)
        # print('--' * 10 + ('domain %d' % domain) + '--' * 10, file=print_to)

        best_model = ''
        best_performance = 0
        best_metrics = {}
        best_epoch = 0
        for epoch in range(opt.nepoch):
            for i, (input, lbl) in enumerate(train_loader):
                input = input.view(-1, MNIST_DIM)
                input = input[:, task_permutations[domain]].to(device)
                model.train()

                model.zero_grad()
                out = model(input)
                loss = criterion(out, lbl.squeeze(0))
                loss.backward()

                # clip_grad_norm_(model.parameters(), 5)
                optim.step()

                utils.progress_bar(i / len(train_loader), loss.item(), epoch)

                if (i + 1) % int(1 / 4 * len(train_loader)) == 0:
                    # valid
                    accurracy, precision, recall, f1 = \
                        valid_mnist(model, valid_loader, task_permutations[domain],device)
                    performance = {'accuracy':accurracy,
                                   'precision':precision,
                                   'recall':recall,
                                   'f1':f1}

                    if performance[opt.metric] > best_performance:
                        print('\r')
                        print(
                            '{\'Epoch\':%d, \'Domain\':%d, \'Format\':\'a/p/r/f\', \'Metrics\':[%4f, %4f, %4f, %4f]}' %
                            (epoch, domain, accurracy, precision, recall, f1))

                        # save model
                        basename = "up-to-domain-{}-epoch-{}".format(domain, epoch)
                        model_fname = basename + ".model"
                        torch.save(model.state_dict(), model_fname)

                        best_performance = performance[opt.metric]
                        best_model = model_fname
                        best_metrics[domain] = (accurracy, precision, recall, f1)
                        best_epoch = epoch

                        # valid the rest
                        for d in range(opt.ndomains):
                            if d == domain:
                                continue

                            accurracy, precision, recall, f1 =\
                                valid_mnist(model, valid_loader, task_permutations[d],device)

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

def train_ll_mnist(model, dataloaders, opt, optim):

    domains = range(opt.ndomains)

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    task_permutations = []
    for _ in domains:
        task_permutations.append(np.random.permutation(784))

    for domain in domains:
        criterion = nn.CrossEntropyLoss()
        train_domain_mnist(model, dataloaders,
                     opt, domain, task_permutations, criterion, optim, device)
