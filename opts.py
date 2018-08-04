import argparse
from macros import *
import os

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=300)
    group.add_argument('-hdim', type=int, default=300)
    group.add_argument('-dropout', type=float, default=0.3)
    # group.add_argument('-pretrain', type=str, default="glove.6B.300d")
    group.add_argument('-pretrain', type=str, default=None)

def train_opts(parser):
    group = parser.add_argument_group('train')
    # group.add_argument('-folder', type=str, default=CHEN)
    # group.add_argument('-ftrain', type=str, default=os.path.join(CHEN, '0.train'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(CHEN, '0.valid'))
    group.add_argument('-ftrain', type=str, default=os.path.join(CHEN, 'unify.train'))
    group.add_argument('-fvalid', type=str, default=os.path.join(CHEN, 'unify.valid'))
    group.add_argument('-ftest', type=str, default=os.path.join(CHEN, 'unify.test'))
    group.add_argument('-finfo', type=str, default=os.path.join(CHEN, 'info'))
    group.add_argument('-bsz', type=int, default=4)
    group.add_argument('-min_freq', type=int, default=5)
    group.add_argument('-nepoch', type=int, default=10)
    group.add_argument('-save_per', type=int, default=5)
    group.add_argument('-name', type=str, default='bigru')
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-lr', type=float, default=1e-3)
    # group.add_argument('-lm_coef', type=float, default=0.5)
    group.add_argument('-wdecay', type=float, default=0)