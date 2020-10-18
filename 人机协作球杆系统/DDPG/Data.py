import numpy as np
import matplotlib.pyplot as plt

font1 = {'family': 'Times new Roman', 'size': 15, }
font2 = {'family': 'FangSong', 'size': 15, }

state_action1 = np.loadtxt('D:/Program Files/Pycharm/Projects/基于强化学习的球杆系统人机协作控制/DDPG/person1 data/state_action.txt')
state_action1 = state_action1[0: 160]
length1 = len(state_action1[:, 3]) # dert_x x x_dot thita thita_dot

state_action2 = np.loadtxt('D:/Program Files/Pycharm/Projects/基于强化学习的球杆系统人机协作控制/DDPG/person2 data//state_action.txt')
state_action2 = state_action2[0: 160]
length2 = len(state_action2[:, 3]) # dert_x x x_dot thita thita_dot

plt.figure(figsize=(10, 5))
plt.plot(np.arange(length1), state_action1[:, 1], label='志愿者1', linestyle='-')
plt.plot(np.arange(length2), state_action2[:, 1], label='志愿者2', linestyle='--')
plt.legend(loc='upper right', prop=font2, frameon=False)
plt.xlabel('step', fontdict=font1)
plt.ylabel('小球位置误差/$m$', fontdict=font2)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(length1), state_action1[:, 3] / np.pi * 180, label='志愿者1', linestyle='-')
plt.plot(np.arange(length2), state_action2[:, 3] / np.pi * 180, label='志愿者2', linestyle='--')
plt.legend(loc='upper right', prop=font2, frameon=False)
plt.xlabel('step', fontdict=font1)
plt.ylabel('长杆角度/$°$', fontdict=font2)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(length1), state_action1[:, 4], label='志愿者1', linestyle='-')
plt.plot(np.arange(length2), state_action2[:, 4], label='志愿者2', linestyle='--')
plt.legend(loc='upper right', prop=font2, frameon=False)
plt.xlabel('step', fontdict=font1)
plt.ylabel('长杆角速度/$°*s^{-1}$', fontdict=font2)
plt.show()
