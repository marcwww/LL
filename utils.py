import numpy as np
from torch.nn.init import xavier_uniform_
from collections import defaultdict

def grad_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def balance_bias(train_iter):

    nlbls = defaultdict(int)
    for batch in train_iter:
        for lbl in batch.lbl.squeeze(0):
            nlbls[lbl.item()] += 1

    res = []
    for i in range(len(nlbls)):
        res.append(1/nlbls[i])

    res = np.array(res)
    res /= res.sum()

    return res

def shift_matrix(n):
    W_up = np.eye(n)
    for i in range(n-1):
        W_up[i,:] = W_up[i+1,:]
    W_up[n-1,:] *= 0
    W_down = np.eye(n)
    for i in range(n-1,0,-1):
        W_down[i,:] = W_down[i-1,:]
    W_down[0,:] *= 0
    return W_up,W_down

def avg_vector(i, n):
    V = np.zeros(n)
    V[:i+1] = 1/(i+1)
    return V

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)

def progress_bar(percent, last_loss, epoch):
    """Prints the progress until the next report."""
    fill = int(percent * 40)
    print("\r[{}{}]: {:.4f}/epoch {:d} (Loss: {:.4f} )".format(
        "=" * fill,
        " " * (40 - fill),
        percent,
        epoch,
        last_loss),
        end='')