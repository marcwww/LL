import torch
from torch import nn
import torch.nn.functional as F
from macros import *
import numpy as np
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence,\
    pack_padded_sequence
import utils
import training_cv
from sklearn.metrics import f1_score, \
    precision_score, \
    recall_score, \
    accuracy_score
import collections

class BiRNN(nn.Module):

    def __init__(self, voc_size, edim, hdim, dropout, padding_idx, nclasses):
        super(BiRNN, self).__init__()

        self.voc_size = voc_size
        self.edim = edim
        self.hdim = hdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                      padding_idx=padding_idx)

        self.rnn = nn.GRU(edim, hdim // 2 ,bidirectional=True,
                          dropout=dropout)
        self.toProbs = nn.Sequential(nn.Linear(hdim, nclasses),
                                    nn.LogSoftmax())

        self.with_lm = False

    def forward(self, inputs):
        seq_len, bsz = inputs.shape
        embs = self.embedding(inputs)
        mask = inputs.data.eq(self.padding_idx)
        input_lens = seq_len - mask.sum(dim=0)

        embs_p = pack_padded_sequence(embs, input_lens)
        # hidden: (2, bsz, hdim/2)
        outputs_p, hidden = self.rnn(embs_p)

        # outputs: (seq_len, bsz, hdim)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        return self.toProbs(hidden)

class Attention(nn.Module):

    def __init__(self, hdim):
        super(Attention, self).__init__()
        self.hdim = hdim
        self.generator = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, 1),
            # nn.Softmax(dim=0)
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs, mask):
        a_raw = self.generator(inputs)
        a_raw.masked_fill_(mask.unsqueeze(-1), -float('inf'))
        a = self.softmax(a_raw)
        return (inputs * a).sum(dim=0)

class RNNAtteion(nn.Module):

    def __init__(self, voc_size, edim, hdim, dropout, padding_idx, nclasses):
        super(RNNAtteion, self).__init__()

        self.voc_size = voc_size
        self.edim = edim
        self.hdim = hdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                      padding_idx=padding_idx)

        self.embedding.weight.requires_grad=False

        self.rnn = nn.GRU(edim, hdim,
                          dropout=dropout)
        self.toProbs = nn.Sequential(nn.Linear(hdim, nclasses),
                                    nn.LogSoftmax())
        self.attention = Attention(hdim)

        self.with_lm = False

    def forward(self, inputs):
        seq_len, bsz = inputs.shape
        embs = self.embedding(inputs)
        mask = inputs.data.eq(self.padding_idx)
        input_lens = seq_len - mask.sum(dim=0)

        embs_p = pack_padded_sequence(embs, input_lens)
        # hidden: (bsz, hdim)
        outputs_p, hidden = self.rnn(embs_p)
        outputs, output_lens = pad_packed_sequence(outputs_p)

        hidden = self.attention(outputs, mask)
        # hidden = hidden.squeeze(0)

        # outputs: (seq_len, bsz, hdim)
        return self.toProbs(hidden)

class RNNAtteionLM(nn.Module):

    def __init__(self, voc_size, edim, hdim, dropout, padding_idx, nclasses):
        super(RNNAtteionLM, self).__init__()

        self.voc_size = voc_size
        self.edim = edim
        self.hdim = hdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                      padding_idx=padding_idx)

        self.rnn = nn.GRU(edim, hdim,
                          dropout=dropout)
        self.toProbs = nn.Sequential(nn.Linear(hdim, nclasses),
                                    nn.LogSoftmax())
        self.attention = Attention(hdim)

        self.with_lm = True

    def forward(self, inputs):
        seq_len, bsz = inputs.shape
        embs = self.embedding(inputs)
        mask = inputs.data.eq(self.padding_idx)
        input_lens = seq_len - mask.sum(dim=0)

        embs_p = pack_padded_sequence(embs, input_lens)
        # hidden: (bsz, hdim)
        outputs_p, hidden = self.rnn(embs_p)
        outputs, output_lens = pad_packed_sequence(outputs_p)

        hidden = self.attention(outputs, mask)
        # hidden = hidden.squeeze(0)

        we_T = self.embedding.weight.transpose(0, 1)
        logits = torch.matmul(outputs, we_T)

        # outputs: (seq_len, bsz, hdim)
        return self.toProbs(hidden), logits

class MaxPooling(nn.Module):

    def __init__(self, voc_size, edim, hdim, dropout, padding_idx, nclasses):
        super(MaxPooling, self).__init__()

        self.voc_size = voc_size
        self.edim = edim
        self.hdim = hdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                      padding_idx=padding_idx)
        self.linear = nn.Linear(edim, hdim)
        self.dropout = nn.Dropout(p=dropout)
        self.toProbs = nn.Sequential(nn.Linear(hdim, nclasses),
                                     nn.LogSoftmax())

        self.with_lm = False

    def forward(self, inputs):
        seq_len, bsz = inputs.shape
        embs = self.embedding(inputs)
        mask = inputs.data.eq(self.padding_idx)
        mask = mask.unsqueeze(-1).expand_as(embs)

        embs = self.dropout(embs)
        embs_affine = self.linear(embs)
        embs_affine.masked_fill_(mask, -float('inf'))
        h, _ = torch.max(embs_affine, dim=0, keepdim=False)

        return self.toProbs(h)

class AvgPooling(nn.Module):

    def __init__(self, voc_size, edim, hdim, dropout, padding_idx, nclasses):
        super(AvgPooling, self).__init__()

        self.voc_size = voc_size
        self.edim = edim
        self.hdim = hdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                      padding_idx=padding_idx)
        self.linear = nn.Linear(edim, hdim)
        self.dropout = nn.Dropout(p=dropout)
        self.toProbs = nn.Sequential(nn.Linear(hdim, nclasses),
                                     nn.LogSoftmax())

        self.with_lm = False

    def forward(self, inputs):
        seq_len, bsz = inputs.shape
        embs = self.embedding(inputs)
        mask = inputs.data.eq(self.padding_idx)
        input_lens = seq_len - mask.sum(dim=0)
        mask = mask.unsqueeze(-1).expand_as(embs)

        embs = self.dropout(embs)
        embs_affine = self.linear(embs)
        # embs_affine: (seq_len, bsz, hdim)
        embs_affine.masked_fill_(mask, 0)

        # h: (bsz, hdim)
        hiddens = torch.sum(embs_affine, dim=0, keepdim=False)
        for i in range(bsz):
            hiddens[i]/=input_lens[i].item()

        return self.toProbs(hiddens)

class MLP(nn.Module):

    def __init__(self, idim, nclasses):
        super(MLP, self).__init__()
        self.hdim = idim
        self.generator = nn.Sequential(nn.Linear(idim, nclasses))

    def forward(self, input):
        res = self.generator(input)
        return res

class MLP2Layers(nn.Module):

    def __init__(self, idim, nclasses):
        super(MLP2Layers, self).__init__()
        self.hdim = idim
        self.layer1 = nn.Sequential(nn.Linear(idim, idim),
                                        nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(idim, nclasses))

    def forward(self, input):
        res = self.layer2(self.layer1(input))
        return res

class BaseMemory(nn.Module):

    def __init__(self, capacity, xdim):
        super(BaseMemory, self).__init__()
        self.capacity = capacity
        self.xdim = xdim
        self.mems_x = nn.Parameter(torch.Tensor(capacity, xdim),
                                 requires_grad=False)
        self.mems_y = nn.Parameter(torch.LongTensor(capacity),
                                   requires_grad=False)

    def fetch(self, *input):
        raise NotImplementedError

    def add(self, *input):
        raise NotImplementedError

class RandomMemory(BaseMemory):

    def __init__(self, capacity, xdim):
        super(RandomMemory, self).__init__(capacity, xdim)
        self.ptr = 0
        self.is_full = False

    def fetch(self, inputs):
        # input: (bsz, hdim)
        bsz = inputs.shape[0]
        res_x = []
        res_y = []
        length = self.capacity if self.is_full else self.ptr
        if length == 0:
            return None, None

        for i in np.random.choice(length, bsz):
            res_x.append(self.mems_x[i].unsqueeze(0))
            res_y.append(self.mems_y[i].unsqueeze(0))

        return torch.cat(res_x, dim=0), \
               torch.cat(res_y, dim=0)

    def add(self, inputs, lbls):
        for input, lbl in zip(inputs, lbls):
            self.mems_x[self.ptr] = input.data
            self.mems_y[self.ptr] = lbl
            self.ptr += 1
            if self.ptr >= self.capacity:
                self.is_full = True

            self.ptr %= self.capacity

    def all_content(self):
        return self.mems_x[:self.ptr], \
               self.mems_y[:self.ptr]

# Random memory MLP
class RAMMLP(MLP):

    def __init__(self, idim, nclasses, capacity,
                 criterion, bsz_sampling):

        super(RAMMLP, self).__init__(idim, nclasses)
        self.mem = RandomMemory(capacity, idim)
        self.nsteps = 0
        self.criterion = criterion
        self.bsz_sampling = bsz_sampling

    def adapt(self, indices, inputs, lbls):
        context_x, context_y = self.mem.fetch(inputs)

        indices_sampled = np.random.choice(len(inputs),
                                   self.bsz_sampling)
        indices_sampled = list(set(indices_sampled))

        self.mem.add(inputs[indices_sampled],
                     lbls[indices_sampled])

        if context_x is not None and\
                context_x is not None:
            out = self.forward(torch.cat([inputs, context_x], dim=0))
            lbl = torch.cat([lbls, context_y], dim=0)
        else:
            out = self.forward(inputs)
            lbl = lbls
        loss = self.criterion(out, lbl)
        self.nsteps +=1
        return loss

class MbPAMemory(BaseMemory):

    def __init__(self, capacity, xdim):
        super(MbPAMemory, self).__init__(capacity, xdim)
        self.ptr = 0
        self.is_full = False

    def fetch(self, inputs):
        pass

    def add(self, inputs, lbls):
        for input, lbl in zip(inputs, lbls):
            self.mems_x[self.ptr] = input
            self.mems_y[self.ptr] = lbl
            self.ptr += 1
            if self.ptr >= self.capacity:
                self.is_full = True

            self.ptr %= self.capacity

class MbPAMLP(MLP):

    def __init__(self, idim, nclasses, capacity,
                 criterion, add_per, device):
        super(MbPAMLP, self).__init__(idim, nclasses)
        self.idim = idim
        self.nclasses = nclasses
        self.mem = RandomMemory(capacity, idim)
        self.criterion = criterion
        self.nsteps = 0
        self.add_per = add_per
        self.epsilon = 1e-4
        self.update_steps = 10
        self.lr = 1e-1
        self.lambda_cache = 0.15
        self.lambda_mbpa = 0.1
        self.K = 128
        self.alpha_m = 5
        self.device = device

    def adapt(self, inputs, lbls):

        if self.nsteps % self.add_per == 0:
            self.mem.add(inputs, lbls)
        if self.nsteps % self.add_per == 0:
            self.mem.add(inputs, lbls)

        out = self.generator(inputs)
        loss = self.criterion(out, lbls.squeeze(0))
        self.nsteps += 1

        return loss

    def _copy_parameters(self):
        res = {}
        params = super(MbPAMLP, self).named_parameters()
        for name, param in params:
            if param.requires_grad:
                res[name] = param.data

        return res

    def _new_mlp(self):
        new_model = MLP(self.idim, self.nclasses)
        params = dict(new_model.named_parameters())
        for name, param in self.named_parameters():
            if param.requires_grad:
                params[name].data.copy_(param.data)

        return new_model.to(self.device)

    def _dis_parameters(self, model_base, model):
        res = 0
        params_base = dict(model_base.named_parameters())
        params = model.named_parameters()

        for name, param in params:
            if param.requires_grad:
                res += torch.norm(param - params_base[name].data, p=2)

        res = torch.pow(res, exponent=2)/(2*self.alpha_m)
        return res

    def _restore_paramaters(self, params_origin):
        cur_params = dict(self.named_parameters())
        for name, param in params_origin.items():
            cur_params[name].data.copy_(param.data)

    def forward(self, input, valid_loader, task_permutation, deep_test, device):

        torch.set_grad_enabled(True)

        tester = self._new_mlp()
        # optimizer = optim.SGD(tester.parameters(),
        #                            lr=self.lr)
        optimizer = optim.Adam(tester.parameters(),
                                   lr=self.lr)
        bsz, _ = input.shape

        mem = self.mem.mems_x if self.mem.is_full \
            else self.mem.mems_x[:self.mem.ptr]
        lbl = self.mem.mems_y if self.mem.is_full \
            else self.mem.mems_y[:self.mem.ptr]

        mem_expanded = mem.unsqueeze(1).\
            expand(mem.shape[0], bsz, mem.shape[1])
        dis_sq = torch.pow(torch.norm(mem_expanded - input, p=2,
                                      dim=-1),
                           exponent=2)
        kern_val = 1/(self.epsilon + dis_sq)

        top_vals, idx = torch.topk(kern_val,
                                   k=min(self.K, kern_val.shape[0]), dim=0)
        top_vals /= top_vals.sum(dim=0)

        mem = mem[idx]
        lbl = lbl[idx]

        for step_idx in range(100 if deep_test else self.update_steps):
            tester.zero_grad()
            tester.train()
            out = tester(mem)
            out = out.view(-1, out.shape[-1])
            lbl = lbl.view(-1)
            # posterior = F.cross_entropy(out, lbl.squeeze(0))
            # loss = posterior
            posterior = F.cross_entropy(out, lbl.squeeze(0), reduce=False)
            posterior = posterior.view(-1, bsz)
            context_loss = (top_vals * posterior).sum(dim=0)
            context_loss = context_loss.sum()/bsz
            paramDis_loss = self._dis_parameters(model_base=self,
                                                 model=tester)

            # losses = paramDis_loss + context_loss
            # loss = losses.sum() / bsz
            loss = (context_loss + paramDis_loss)/2
            # loss = context_loss
            loss.backward()
            optimizer.step()

            if deep_test:
                tester.eval()
                pred_lst = []
                true_lst = []

                with torch.no_grad():
                    for batch_idx, (input_test, lbl_test) in enumerate(valid_loader):
                        input_test = input_test.view(-1, MNIST_DIM)
                        input_test = input_test[:, task_permutation].to(device)
                        lbl_test = lbl_test.squeeze(0).to(device)
                        # probs: (bsz, 3)

                        out = tester(input_test)

                        pred = out.max(dim=1)[1].cpu().numpy()
                        lbl_test = lbl_test.cpu().numpy()
                        pred_lst.extend(pred)
                        true_lst.extend(lbl_test)

                accurracy = accuracy_score(true_lst, pred_lst)
                precision = precision_score(true_lst, pred_lst, average='macro')
                recall = recall_score(true_lst, pred_lst, average='macro')
                f1 = f1_score(true_lst, pred_lst, average='macro')

                # if step_idx >= 15:
                #     optimizer.param_groups[0]['lr'] = self.lr/10

                print('deep_test %d/%d:' % (step_idx,self.update_steps),
                      context_loss.item(),
                      paramDis_loss.item(),
                      f1)

        out = tester(input)

        return out

class MbPAMLP2Layers(MLP2Layers):

    def __init__(self, idim, nclasses, capacity,
                 criterion, add_per, device):
        super(MbPAMLP2Layers, self).__init__(idim, nclasses)
        self.idim = idim
        self.nclasses = nclasses
        self.mem = RandomMemory(capacity, idim)
        self.criterion = criterion
        self.nsteps = 0
        self.add_per = add_per
        self.epsilon = 1e-4
        self.update_steps = 41
        self.lr = 1e-3
        self.lambda_cache = 0.15
        self.lambda_mbpa = 0.1
        self.K = 128
        self.alpha_m = 10
        self.device = device

    def adapt(self, inputs, lbls):

        embeddings = self.layer1(inputs)

        if self.nsteps % self.add_per == 0:
            self.mem.add(embeddings, lbls)
        if self.nsteps % self.add_per == 0:
            self.mem.add(embeddings, lbls)

        out = self.layer2(embeddings)
        loss = self.criterion(out, lbls.squeeze(0))
        self.nsteps += 1

        return loss

    def _copy_parameters(self):
        res = {}
        params = super(MbPAMLP2Layers, self).named_parameters()
        for name, param in params:
            if param.requires_grad:
                res[name] = param.data

        return res

    def _new_mlp(self):
        new_model = MLP2Layers(self.idim, self.nclasses)
        params = dict(new_model.named_parameters())
        for name, param in self.named_parameters():
            if param.requires_grad and name in params.keys():
                params[name].data.copy_(param.data)

        return new_model.to(self.device)

    def _dis_parameters(self, model_base, model):
        res = 0
        params_base = dict(model_base.named_parameters())
        params = model.named_parameters()

        for name, param in params:
            if param.requires_grad:
                res += torch.norm(param - params_base[name].data, p=2)

        res = torch.pow(res, exponent=2)/(2*self.alpha_m)
        return res

    def _restore_paramaters(self, params_origin):
        cur_params = dict(self.named_parameters())
        for name, param in params_origin.items():
            cur_params[name].data.copy_(param.data)

    def forward(self, input, valid_loader, task_permutation, deep_test, device):

        torch.set_grad_enabled(True)

        tester = self._new_mlp()
        # optimizer = optim.SGD(tester.parameters(),
        #                            lr=self.lr)
        optimizer = optim.Adam(tester.parameters(),
                                   lr=self.lr)
        bsz, _ = input.shape

        mem = self.mem.mems_x if self.mem.is_full \
            else self.mem.mems_x[:self.mem.ptr]
        lbl = self.mem.mems_y if self.mem.is_full \
            else self.mem.mems_y[:self.mem.ptr]

        mem_expanded = mem.unsqueeze(1).\
            expand(mem.shape[0], bsz, mem.shape[1])
        embedding = self.layer1(input)
        dis_sq = torch.pow(torch.norm(mem_expanded - embedding.data, p=2,
                                      dim=-1),
                           exponent=2)
        kern_val = 1/(self.epsilon + dis_sq)

        top_vals, idx = torch.topk(kern_val,
                                   k=min(self.K, kern_val.shape[0]), dim=0)
        top_vals /= top_vals.sum(dim=0)

        mem = mem[idx]
        lbl = lbl[idx]

        for step_idx in range(100 if deep_test else self.update_steps):
            tester.zero_grad()
            tester.train()
            out = tester.layer2(mem)
            out = out.view(-1, out.shape[-1])
            lbl = lbl.view(-1)
            # posterior = F.cross_entropy(out, lbl.squeeze(0))
            # loss = posterior
            posterior = F.cross_entropy(out, lbl.squeeze(0), reduce=False)
            posterior = posterior.view(-1, bsz)
            context_loss = (top_vals * posterior).sum(dim=0)
            context_loss = context_loss.sum()/bsz
            paramDis_loss = self._dis_parameters(model_base=self,
                                                 model=tester)

            # losses = paramDis_loss + context_loss
            # loss = losses.sum() / bsz
            loss = (context_loss + paramDis_loss)/2
            # loss = context_loss
            loss.backward()
            optimizer.step()

            # if deep_test:
            #     tester.eval()
            #     pred_lst = []
            #     true_lst = []
            #
            #     with torch.no_grad():
            #         for batch_idx, (input_test, lbl_test) in enumerate(valid_loader):
            #             input_test = input_test.view(-1, MNIST_DIM)
            #             input_test = input_test[:, task_permutation].to(device)
            #             lbl_test = lbl_test.squeeze(0).to(device)
            #             # probs: (bsz, 3)
            #
            #             out = tester(input_test)
            #
            #             pred = out.max(dim=1)[1].cpu().numpy()
            #             lbl_test = lbl_test.cpu().numpy()
            #             pred_lst.extend(pred)
            #             true_lst.extend(lbl_test)
            #
            #     accurracy = accuracy_score(true_lst, pred_lst)
            #     precision = precision_score(true_lst, pred_lst, average='macro')
            #     recall = recall_score(true_lst, pred_lst, average='macro')
            #     f1 = f1_score(true_lst, pred_lst, average='macro')
            #
            #     # if step_idx >= 15:
            #     #     optimizer.param_groups[0]['lr'] = self.lr/10
            #
            #     print('deep_test %d/%d:' % (step_idx,self.update_steps),
            #           context_loss.item(),
            #           paramDis_loss.item(),
            #           f1)

        out = tester(input)

        return out

# Gradient importance memory
class GradientMemory(BaseMemory):

    def __init__(self, capacity, xdim, retain_ratio):
        super(GradientMemory, self).__init__(capacity, xdim)
        # gradient norms
        self.mems_g = nn.Parameter(torch.Tensor(capacity),
                                   requires_grad=False)
        self.mems_i = nn.Parameter(torch.LongTensor(capacity),
                                   requires_grad=False)
        self.ptr = 0
        # pointer for last domian
        self.ld_ptr = 0
        self.is_full = False
        self.retain_ratio = retain_ratio

    def fetch(self, inputs):
        # input: (bsz, hdim)
        bsz = inputs.shape[0]
        res_x = []
        res_y = []
        res_g = []
        res_i = []
        length = self.capacity if self.is_full else self.ptr
        if length == 0:
            return None, None, None, None

        for i in np.random.choice(length, bsz):
            res_x.append(self.mems_x[i].unsqueeze(0))
            res_y.append(self.mems_y[i].unsqueeze(0))
            res_g.append(self.mems_g[i].unsqueeze(0))
            res_i.append(self.mems_i[i].unsqueeze(0))

        return torch.cat(res_i, dim=0), \
               torch.cat(res_x, dim=0), \
               torch.cat(res_y, dim=0), \
               torch.cat(res_g, dim=0)

    def add(self, indices, inputs, lbls, gnorms):
        for index, input, lbl, gnorm in \
                zip(indices, inputs, lbls, gnorms):
            self.mems_i[self.ptr] = int(index)
            self.mems_x[self.ptr] = input.data
            self.mems_y[self.ptr] = lbl
            self.mems_g[self.ptr] = gnorm

            self.ptr += 1
            if self.ptr >= self.capacity:
                self.is_full = True

            self.ptr %= self.capacity

    def _topk(self, Is, Xs, Gs, Ys, K):
        top_gnorms, top_idices = torch.topk(Gs, k=K)
        return Is[top_idices], \
               Xs[top_idices], \
               Ys[top_idices], \
               Gs[top_idices]

    # called when done training on a domain
    def trim(self):

        print('before trimming:', self.ld_ptr, self.ptr)
        if self.ld_ptr< self.ptr:
            Is = self.mems_i[self.ld_ptr:self.ptr]
            Xs = self.mems_x[self.ld_ptr:self.ptr]
            Ys = self.mems_y[self.ld_ptr:self.ptr]
            Gs = self.mems_g[self.ld_ptr:self.ptr]
        else:
            Is = torch.cat([self.mems_i[self.ld_ptr:],
                            self.mems_i[:self.ptr]], dim=0)
            Xs = torch.cat([self.mems_x[self.ld_ptr:],
                            self.mems_x[:self.ptr]], dim=0)
            Ys = torch.cat([self.mems_y[self.ld_ptr:],
                            self.mems_y[:self.ptr]], dim=0)
            Gs = torch.cat([self.mems_g[self.ld_ptr:],
                            self.mems_g[:self.ptr]], dim=0)

        K = int(len(Gs) * self.retain_ratio)

        hist_y = collections.defaultdict(int)
        classes = collections.defaultdict()

        for i, x, y, g in zip(Is, Xs, Ys, Gs):
            hist_y[y.item()] += 1
            if y.item() not in classes.keys():
                classes[y.item()]={'i':[],'x':[],'y':[],'g':[]}

            classes[y.item()]['i'].append(i.unsqueeze(0))
            classes[y.item()]['x'].append(x.unsqueeze(0))
            classes[y.item()]['y'].append(y.unsqueeze(0))
            classes[y.item()]['g'].append(g.unsqueeze(0))

        hist_y_min = hist_y[min(hist_y, key=lambda y: hist_y[y])]
        nclasses = len(hist_y)
        K_class = min(int(K / nclasses), hist_y_min)

        res_i = []
        res_x = []
        res_y = []
        res_g = []
        for y in classes.keys():
            Is = torch.cat(classes[y]['i'])
            Xs = torch.cat(classes[y]['x'])
            Ys = torch.cat(classes[y]['y'])
            Gs = torch.cat(classes[y]['g'])
            Is, Xs, Ys, Gs = self._topk(Is, Xs, Gs, Ys, K_class)
            res_i.append(Is)
            res_x.append(Xs)
            res_y.append(Ys)
            res_g.append(Gs)

        res_i = torch.cat(res_i, dim=0)
        res_x = torch.cat(res_x, dim=0)
        res_y = torch.cat(res_y, dim=0)
        res_g = torch.cat(res_g, dim=0)

        num = K_class * nclasses
        self.mems_i[self.ld_ptr:self.ld_ptr + num] = res_i
        self.mems_x[self.ld_ptr:self.ld_ptr + num] = res_x
        self.mems_y[self.ld_ptr:self.ld_ptr + num] = res_y
        self.mems_g[self.ld_ptr:self.ld_ptr + num] = res_g
        self.ptr = self.ld_ptr + num

        if self.ptr < self.capacity:
            self.is_full = False

        self.ld_ptr = self.ptr
        print('after trimming:', self.ld_ptr, self.ptr)

    def all_content(self):
        return self.mems_i[:self.ptr],\
               self.mems_x[:self.ptr],\
               self.mems_y[:self.ptr],\
               self.mems_g[:self.ptr],

class GNIMLP(MLP):

    def __init__(self, idim, nclasses, capacity,
                 criterion, add_per, retain_ratio,
                 device):

        super(GNIMLP, self).__init__(idim, nclasses)
        self.mem = GradientMemory(capacity, idim, retain_ratio)
        self.idim = idim
        self.nclasses = nclasses
        self.nsteps = 0
        self.criterion = criterion
        self.add_per = add_per
        self.device = device
        self.baby_mlp = self._fork()

    def _fork(self):
        new_model = MLP(self.idim, self.nclasses)
        new_model_params = dict(new_model.named_parameters())
        for name, param in self.named_parameters():
            if param.requires_grad and name in new_model_params.keys():
                new_model_params[name].data.copy_(param.data)

        return new_model.to(self.device)

    def adapt(self, indices, inputs, lbls):
        context_i, context_x, context_y, context_g = \
            self.mem.fetch(inputs)

        out = self.baby_mlp(inputs)
        self.criterion.reduce = False
        loss_out = self.criterion(out, lbls.squeeze(0))
        self.criterion.reduce = True
        gnorms = []
        for loss in loss_out:
            self.baby_mlp.zero_grad()
            loss.backward(retain_graph=True)
            gnorm = utils.grad_norm(self.baby_mlp.parameters())
            gnorms.append(gnorm)
        self.baby_mlp.zero_grad()

        if self.nsteps % self.add_per == 0:
            self.mem.add(indices, inputs, lbls, gnorms)

        if context_x is not None and \
                context_x is not None:
            out = self.forward(torch.cat([inputs, context_x], dim=0))
            lbl = torch.cat([lbls, context_y], dim=0)
        else:
            out = self.forward(inputs)
            lbl = lbls

        loss = self.criterion(out, lbl.squeeze(0))
        self.nsteps += 1

        return loss

    def trim(self):
        self.mem.trim()

        Is, Xs, Ys, Gs = self.mem.all_content()
        utils.draw_dist(Ys.data.numpy(), bins=10)

# Gradient sampling memory
class GSMLP(MLP):

    def __init__(self, idim, nclasses, capacity,
                 criterion, bsz_sampling):

        super(GSMLP, self).__init__(idim, nclasses)
        self.mem = RandomMemory(capacity, idim)
        self.nsteps = 0
        self.criterion = criterion
        self.bsz_sampling = bsz_sampling


    def adapt(self, indices, inputs, lbls):
        context_x, context_y = self.mem.fetch(inputs)

        out = self.forward(inputs)
        self.criterion.reduce = False
        loss_out = self.criterion(out, lbls.squeeze(0))
        self.criterion.reduce = True
        gnorms = []
        for loss in loss_out:
            self.zero_grad()
            loss.backward(retain_graph=True)
            gnorm = utils.grad_norm(self.parameters())
            gnorms.append(gnorm.item())
        self.zero_grad()
        gnorms = np.array(gnorms)
        gnorms /= gnorms.sum()
        indices_sampled = np.random.choice(len(gnorms), self.bsz_sampling, p=gnorms)
        indices_sampled = list(set(indices_sampled))
        self.mem.add(inputs[indices_sampled], lbls[indices_sampled])

        if context_x is not None and\
                context_x is not None:
            out = self.forward(torch.cat([inputs, context_x], dim=0))
            lbl = torch.cat([lbls, context_y], dim=0)
        else:
            out = self.forward(inputs)
            lbl = lbls
        loss = self.criterion(out, lbl.squeeze(0))
        self.nsteps +=1
        return loss






















