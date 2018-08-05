import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from macros import *

from glob import glob
import json
import os
import sys
import demjson

flog = 'rnn_atten.log'
with open(os.path.join(RES, flog), 'r') as f:
    lines = f.readlines()
    maps = []
    domain_pos = []

    for i, line in enumerate(lines):
        one_map = demjson.decode(line)
        if len(one_map) == 1:
            domain_pos.append(i)
        else:
            maps.append(one_map)

    domain_f1s = {}
    for i in range(len(domain_pos) - 1):
        domain_f1s[i] = []
        begin = domain_pos[i]
        end = domain_pos[i+1]
        records = maps[begin:end]
        for record in records:
            dom, f1 = record['Domain'], record['Metrics'][-1]
            domain_f1s[dom].append(f1)

    print(domain_f1s)

domain = [9]
# domain = range(20)

for d, f1s in domain_f1s.items():
    if d in domain:
        x = np.array(range(0, len(f1s), 1)) + d * 1
        y = [f1s[i] for i in range(0, len(f1s), 1)]
        plt.plot(x, y, label='d{}'.format(d))

plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(np.arange(0, 20, 1))
plt.legend(loc=0)
plt.show()

#
# loss = history[3]['loss']
# cost = history[3]['cost']
# seq_lengths = history[3]['seq_lengths']
#
# unique_sls = set(seq_lengths)
# all_metric = list(zip(range(1, batch_num + 1), seq_lengths, loss, cost))
#
# fig = plt.figure(figsize=(12, 5))
# plt.ylabel('Cost per sequence (bits)')
# plt.xlabel('Iteration (thousands)')
# plt.title('Training Convergence (Per Sequence Length)', fontsize=16)
#
# for sl in unique_sls:
#     sl_metrics = [i for i in all_metric if i[1] == sl]
#
#     x = [i[0] for i in sl_metrics]
#     y = [i[3] for i in sl_metrics]
#
#     num_pts = len(x) // 50
#     total_pts = num_pts * 50
#
#     x_mean = [i.mean() / 1000 for i in np.split(np.array(x)[:total_pts], num_pts)]
#     y_mean = [i.mean() for i in np.split(np.array(y)[:total_pts], num_pts)]
#
#     plt.plot(x_mean, y_mean, label='Seq-{}'.format(sl))
#
# plt.yticks(np.arange(0, 80, 5))
# plt.legend(loc=0)
# plt.show()