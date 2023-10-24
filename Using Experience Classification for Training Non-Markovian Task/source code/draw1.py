import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inst_num = 10
x = [None]*inst_num
y = [None]*inst_num
y_new_rew = [None]*inst_num
y_new_len = [None]*inst_num
x_new = np.linspace(0, 300000, 200000)


for i in range(inst_num):
    path_rew = "/Users/miaoruixuan/Desktop/data/cartpole_task3/run-TD3_EC_inst1_" + str(
        i + 1) + "-tag-rollout_ep_rew_mean.csv"
    path_len = "/Users/miaoruixuan/Desktop/data/cartpole_task3/run-TD3_EC_inst1_" + str(
        i + 1) + "-tag-rollout_ep_len_mean.csv"
    data = pd.read_csv(path_rew)
    x[i] = data["Step"]
    x[i] = np.array(x[i])
    y[i] = data["Value"]
    y[i] = np.array(y[i])
    y_new_rew[i] = np.interp(x_new, x[i], y[i])
    data = pd.read_csv(path_len)
    x[i] = data["Step"]
    x[i] = np.array(x[i])
    y[i] = data["Value"]
    y[i] = np.array(y[i])
    y_new_len[i] = np.interp(x_new, x[i], y[i])
y_new = np.divide(y_new_rew, y_new_len)
y_new = np.array(y_new)
y_mean = np.mean(y_new, axis=0)
y_max = np.max(y_new, axis=0)
y_min = np.min(y_new, axis=0)
plt.fill_between(x_new, y_max, y_min, alpha=0.3)
plt.plot(x_new, y_mean, label='TD3_EC')

for i in range(inst_num):
    path_rew = "/Users/miaoruixuan/Desktop/data/cartpole_task3/run-TD3_NEC_inst1_" + str(
        i + 1) + "-tag-rollout_ep_rew_mean.csv"
    path_len = "/Users/miaoruixuan/Desktop/data/cartpole_task3/run-TD3_NEC_inst1_" + str(
        i + 1) + "-tag-rollout_ep_len_mean.csv"
    data = pd.read_csv(path_rew)
    x[i] = data["Step"]
    x[i] = np.array(x[i])
    y[i] = data["Value"]
    y[i] = np.array(y[i])
    y_new_rew[i] = np.interp(x_new, x[i], y[i])
    data = pd.read_csv(path_len)
    x[i] = data["Step"]
    x[i] = np.array(x[i])
    y[i] = data["Value"]
    y[i] = np.array(y[i])
    y_new_len[i] = np.interp(x_new, x[i], y[i])
y_new = np.divide(y_new_rew, y_new_len)
y_new = np.array(y_new)
y_mean = np.mean(y_new, axis=0)
y_max = np.max(y_new, axis=0)
y_min = np.min(y_new, axis=0)
plt.fill_between(x_new, y_max, y_min, alpha=0.3 )
plt.plot(x_new, y_mean, label='TD3_NEC')

for i in range(inst_num):
    path_rew = "/Users/miaoruixuan/Desktop/data/cartpole_task3/run-SAC_EC_inst1_" + str(
        i + 1) + "-tag-rollout_ep_rew_mean.csv"
    path_len = "/Users/miaoruixuan/Desktop/data/cartpole_task3/run-SAC_EC_inst1_" + str(
        i + 1) + "-tag-rollout_ep_len_mean.csv"
    data = pd.read_csv(path_rew)
    x[i] = data["Step"]
    x[i] = np.array(x[i])
    y[i] = data["Value"]
    y[i] = np.array(y[i])
    y_new_rew[i] = np.interp(x_new, x[i], y[i])
    data = pd.read_csv(path_len)
    x[i] = data["Step"]
    x[i] = np.array(x[i])
    y[i] = data["Value"]
    y[i] = np.array(y[i])
    y_new_len[i] = np.interp(x_new, x[i], y[i])
y_new = np.divide(y_new_rew, y_new_len)
y_new = np.array(y_new)
y_mean = np.mean(y_new, axis=0)
y_max = np.max(y_new, axis=0)
y_min = np.min(y_new, axis=0)
plt.fill_between(x_new, y_max, y_min, alpha=0.3 )
plt.plot(x_new, y_mean, label='SAC_EC')

for i in range(inst_num):
    path_rew = "/Users/miaoruixuan/Desktop/data/cartpole_task3/run-SAC_NEC_inst1_" + str(
        i + 1) + "-tag-rollout_ep_rew_mean.csv"
    path_len = "/Users/miaoruixuan/Desktop/data/cartpole_task3/run-SAC_NEC_inst1_" + str(
        i + 1) + "-tag-rollout_ep_len_mean.csv"
    data = pd.read_csv(path_rew)
    x[i] = data["Step"]
    x[i] = np.array(x[i])
    y[i] = data["Value"]
    y[i] = np.array(y[i])
    y_new_rew[i] = np.interp(x_new, x[i], y[i])
    data = pd.read_csv(path_len)
    x[i] = data["Step"]
    x[i] = np.array(x[i])
    y[i] = data["Value"]
    y[i] = np.array(y[i])
    y_new_len[i] = np.interp(x_new, x[i], y[i])
y_new = np.divide(y_new_rew, y_new_len)
y_new = np.array(y_new)
y_mean = np.mean(y_new, axis=0)
y_max = np.max(y_new, axis=0)
y_min = np.min(y_new, axis=0)
plt.fill_between(x_new, y_max, y_min, alpha=0.3 )
plt.plot(x_new, y_mean, label='SAC_NEC')
plt.xlabel('Training steps (in thousands)',fontsize=16, fontweight='bold')
plt.ylabel('Avg. rewards per step',fontsize=16, fontweight='bold')
plt.xticks([0, 50000, 100000, 150000, 200000, 250000, 300000], [0, 50, 100, 150, 200, 250, 300])
plt.ylim(-0.1)
plt.title('Task 6', fontsize=16, fontweight='bold')
plt.legend()
plt.savefig(fname="figure/task6_quality.png")
plt.show()