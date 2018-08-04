import torch
from torch import nn
import torch.nn.functional as F
from macros import *
from torch.nn.utils.rnn import pad_packed_sequence,\
    pack_padded_sequence


class BaseRNN(nn.Module):

    def __init__(self, voc_size, edim, hdim, dropout, padding_idx):
        super(BaseRNN, self).__init__()

        self.voc_size = voc_size
        self.edim = edim
        self.hdim = hdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                      padding_idx=padding_idx)

        self.rnn = nn.GRU(edim, hdim // 2 ,bidirectional=True,
                          dropout=dropout)
        self.toProbs = nn.Sequential(nn.Linear(hdim, 3),
                                    nn.LogSoftmax())

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

class BasePooling(nn.Module):

    def __init__(self, voc_size, edim, hdim, dropout, padding_idx):
        super(BasePooling, self).__init__()

        self.voc_size = voc_size
        self.edim = edim
        self.hdim = hdim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(voc_size, edim,
                                      padding_idx=padding_idx)
        self.linear = nn.Linear(hdim, hdim)
        self.dropout = nn.Dropout(p=dropout)
        self.toProbs = nn.Sequential(nn.Linear(hdim, 3),
                                     nn.LogSoftmax())

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





