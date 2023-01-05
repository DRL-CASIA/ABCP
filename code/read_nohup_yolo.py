#!/usr/bin/env python
# -*- coding-utf-8 -*-
# xuer ----time:
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Times New Roman'],
    })
epochs = []
child_flops = []
child_test_loss = []
controller_loss = []
controller_entropy = []
controller_baseline = []
controller_reward = []
controller_reward_baseline = []

with open('nohup.out', 'r') as nohup_lines:
    if_flops = 0
    for line in nohup_lines:
        if 'epoch=' in line:
            if not if_flops:
                flops = (line.split('flops=')[1]).split(' ')[0]
                child_flops.append(float(flops))
                if_flops += 1
        elif 'ctrl_step' in line:
            epochs_now = (line.split('ctrl_step=')[1]).split(' ')[0]
            epochs.append(int(epochs_now))
        elif 'test_epoch_loss' in line:
            test_loss = line.split(':  ')[1].rstrip()
            child_test_loss.append(float(test_loss))
        #     test_loss = (line.split('test_loss=')[1]).split(' ')[0]
        #     child_test_loss.append(float(test_loss))
            # ctr_loss = (line.split('loss=')[1]).split(' ')[0]
            # controller_loss.append(-(float(ctr_loss)))
            # ctr_ent = (line.split('ent=')[1]).split(' ')[0]
            # controller_entropy.append(float(ctr_ent))
            # ctr_bl = (line.split('bl=')[1]).split(' ')[0]
            # controller_baseline.append(float(ctr_bl))
            reward = -float(test_loss) - float(flops) /500000.0
            controller_reward.append(reward)
            # reward_baseline = reward - 0.01 * float(ctr_bl)
            # controller_reward_baseline.append(reward_baseline)
            if_flops = 0



plt.plot(epochs[0:-1], child_test_loss[0:-1], ms=15)
plt.savefig('child_test_loss.tif', bbox_inches='tight')
plt.show()
plt.plot(epochs[0:-1], child_flops[0:-1], ms=15)
plt.savefig('child_flops.tif', bbox_inches='tight')
plt.show()
# plt.plot(epochs[0:-1], controller_loss[0:-1], ms=15)
# plt.savefig('controller_loss.tif', bbox_inches='tight')
# plt.show()
# plt.plot(epochs[0:-1], controller_entropy[0:-1], ms=15)
# plt.savefig('controller_entropy.tif', bbox_inches='tight')
# plt.show()
# plt.plot(epochs[0:-1], controller_baseline[0:-1], ms=15)
# plt.savefig('controller_baseline.tif', bbox_inches='tight')
# plt.show()
plt.plot(epochs[0:-1], controller_reward[0:-1], ms=15)
plt.savefig('controller_reward.tif', bbox_inches='tight')
plt.show()
# plt.plot(epochs[0:-1], controller_reward_baseline[0:-1], ms=15)
# plt.savefig('controller_reward_baseline.tif', bbox_inches='tight')
# plt.show()

test_loss = np.array(child_test_loss)
sort_test_loss = np.argsort(test_loss)
flops = np.array(child_flops)
sort_flops = np.argsort(flops)
reward = np.array(controller_reward)
sort_reward = np.argsort(reward)
print('test_loss: ', sort_test_loss)
print('flops: ', sort_flops)
print('reward:', sort_reward)  # the true index should be added 1
# 200, 267, 180, 227, 229, 302, 260, 228
# 243, 224, 171, 185/ 190, 176, 185, 243
# 314, 309, 195, 448, 188
