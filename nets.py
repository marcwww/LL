import torch
from torch import nn
import torch.nn.functional as F
from macros import *
import numpy as np
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence,\
    pack_padded_sequence
import utils

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
        self.generator = nn.Sequential(
            nn.Linear(idim, nclasses)
        )

    def forward(self, input):
        res = self.generator(input)
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
            self.mems_x[self.ptr] = input
            self.mems_y[self.ptr] = lbl
            self.ptr += 1
            if self.ptr >= self.capacity:
                self.is_full = True

            self.ptr %= self.capacity

# Random memory MLP
class RAMMLP(MLP):

    def __init__(self, idim, nclasses, capacity,
                 criterion, add_per):

        super(RAMMLP, self).__init__(idim, nclasses)
        self.mem = RandomMemory(capacity, idim)
        self.nsteps = 0
        self.criterion = criterion
        self.add_per = add_per

    def adapt(self, inputs, lbls):
        context_x, context_y = self.mem.fetch(inputs)
        if self.nsteps % self.add_per == 0:
            self.mem.add(inputs, lbls)
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
        self.update_steps = 5
        self.lr = 0.15
        self.lambda_cache = 0.15
        self.lambda_mbpa = 0.1
        self.K = 256
        self.alpha_m = 1e-2
        self.device = device

    def adapt(self, inputs, lbls, domain):
        if domain == 0:
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

        res = -1 * torch.pow(res, exponent=2)/(2*self.alpha_m)
        return res

    def _restore_paramaters(self, params_origin):
        cur_params = dict(self.named_parameters())
        for name, param in params_origin.items():
            cur_params[name].data.copy_(param.data)

    def forward(self, input):

        torch.set_grad_enabled(True)

        tester = self._new_mlp()
        optimizer = optim.SGD(tester.parameters(),
                                   lr=self.lr)
        bsz, _ = input.shape

        for _ in range(self.update_steps):
            mem = self.mem.mems_x if self.mem.is_full \
                else self.mem.mems_x[:self.mem.ptr+1]
            lbl = self.mem.mems_y if self.mem.is_full \
                else self.mem.mems_y[:self.mem.ptr+1]

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
            posterior = F.log_softmax(tester.forward(mem),
                                      dim=-1)
            # posterior = torch.log(self.generator(mem[idx]))

            nclasses = posterior.shape[-1]
            lbl = lbl[idx]
            posterior = utils.select_1d(posterior.view(-1, nclasses),
                                        lbl.view(-1),
                                        self.device)
            posterior = posterior.view(-1, bsz)

            context_loss = -1 * (top_vals * posterior).sum(dim=0)
            paramDis_loss = self._dis_parameters(model_base=self,
                                                 model=tester)

            losses = paramDis_loss + context_loss
            loss = losses.sum() / bsz
            loss.backward()
            optimizer.step()

        out = self.generator(input)

        return out

















