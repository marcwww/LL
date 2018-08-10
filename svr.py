import torch
from torchvision import datasets, transforms
import opts
import argparse
import nets
from macros import *
import utils
import numpy as np
from torch import nn
from torch import optim
from sklearn.metrics import f1_score, \
    precision_score, \
    recall_score, \
    accuracy_score
import torch.nn.functional as F
from sklearn.svm import NuSVR
import crash_on_ipy
import time
from sklearn.externals import joblib

def valid(model, valid_loader, task_permutation, device):
    model.eval()
    pred_lst = []
    true_lst = []

    with torch.no_grad():
        for i, (input, lbl) in enumerate(valid_loader):
            input = input.view(-1, MNIST_DIM)
            input = input[:, task_permutation].to(device)
            lbl = lbl.squeeze(0).to(device)
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

def train(model, dataloaders, opt,
                       task_permutation, criterion, optim, device):

    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']

    for epoch in range(opt.nepoch):
        for i, (input, lbl) in enumerate(train_loader):
            input = input.view(-1, MNIST_DIM)
            input = input[:, task_permutation].to(device)
            lbl = lbl.to(device)
            model.train()

            model.zero_grad()
            out = model(input)
            loss = criterion(out, lbl)
            loss.backward()

            # clip_grad_norm_(model.parameters(), 5)
            optim.step()

            utils.progress_bar(i / len(train_loader), loss.item(), epoch)

            if (i + 1) % int(opt.test_per_ratio * len(train_loader)) == 0:
                # valid
                accurracy, precision, recall, f1 = \
                    valid(model, valid_loader, task_permutation, device)

                print('{\'Epoch\':%d, \'Format\':\'a/p/r/f\', '
                      '\'Metrics\':[%4f, %4f, %4f, %4f]}' %
                      (epoch, accurracy, precision, recall, f1))

    # save model
    basename = "mlp"
    model_fname = basename + ".model"
    torch.save(model.state_dict(), model_fname)
    print('saving to', model_fname)

def distill(model, dataloaders, task_permutation, device):
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    X = []
    Y = []
    tar_class = 0

    for i, (inputs, lbls) in enumerate(train_loader):
        inputs = inputs.view(-1, MNIST_DIM)
        inputs = inputs[:, task_permutation].to(device)
        logits = model(inputs)
        probs = F.gumbel_softmax(logits, tau=1)
        for input, prob, lbl in zip(inputs, probs, lbls):
            X.append(input.data.numpy())
            Y.append(prob.data[tar_class].numpy())
            # Y.append(prob.data[lbl.item()].numpy())

    X = np.array(X)
    Y = np.array(Y)
    clf = NuSVR(C=1.0, nu=0.1, max_iter=10)
    t1 = time.time()
    res = clf.fit(X, Y)
    t2 = time.time()
    model_fname = 'distilled_svr.m'
    joblib.dump(clf, model_fname)
    print('saving to', model_fname)
    # print(clf.support_vectors_)
    print(res)
    print(t2-t1)

def load_opt():
    parser = argparse. \
        ArgumentParser(description='main.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)

    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()
    return opt

def load_data():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=opt.bsz, shuffle=True, )
    # **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=opt.bsz, shuffle=True, )

    return {'train':train_loader,
            'valid':valid_loader}

def build_model(opt):
    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)
    model = nets.MLP(opt.idim, opt.nclasses).to(device)
    task_permutation = np.random.permutation(784)
    # task_permutation = np.array(range(784))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                           lr=opt.lr)

    return device, model, task_permutation, criterion, optimizer

def load_mlp(opt, device):
    basename = "mlp"
    model_fname = basename + ".model"
    location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
    model_dict = torch.load(model_fname, map_location=location)
    model = nets.MLP(opt.idim, opt.nclasses).to(device)
    model.load_state_dict(model_dict)
    return model

def load_svr():
    model_fname = 'distilled_svr.m'
    return joblib.load(model_fname)

if __name__ == '__main__':

    # is_training = False
    is_training = True

    # init opt
    opt = load_opt()

    # load data
    dataloaders = load_data()

    # build model
    device, model, task_permutation, criterion, optimizer \
        = build_model(opt)

    # load mlp
    model = load_mlp(opt, device)

    if is_training:

        # train model
        train(model, dataloaders,
              opt, task_permutation,
              criterion, optimizer, device)

        # distll mlp to svr
        distill(model, dataloaders,
                task_permutation,
                device)

    #load svr
    clf = load_svr()
    print(clf)