""" 导入numpy和matplotlib """
import numpy as np
import matplotlib.pyplot as plt
""" 导入sciki-learn的高斯过程模块 """
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
""" 导入joblib存储模块 """
import joblib

""" 创建训练集 """
data_train = np.loadtxt('D:/Program Files/Pycharm/Projects/基于强化学习的球杆系统人机协作控制/GPR/改进版training_data.txt')
# feature_train = data_train[:, :6]  # 数据集前4列作为特征（误差，距离，速度，角度）
# feature_train = feature_train[0:2000]
# label_x = data_train[:, -2]  # 数据集第5列作为x轴速度的标签
# label_x = label_x[0:2000]
# label_z = data_train[:, -1]  # 数据集第6列作为z轴速度的标签
# label_z = label_z[0:2000]

""" 创建测试集 """
data_test = np.loadtxt('D:/Program Files/Pycharm/Projects/基于强化学习的球杆系统人机协作控制/GPR/改进版training_data.txt')
feature_test = data_test[:, :6]
feature_test = feature_test[2000:2100]  # 取数据作为测试集

"""实际速度 """
actual_x = data_train[:, -2]
actual_x = actual_x[2000:2100]
actual_z = data_train[:, -1]
actual_z = actual_z[2000:2100]

""" 创建核函数 """
# kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))

# """ 创建高斯过程回归模型 """
# reg_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.01)
# reg_z = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

# """ 训练x轴超参数 """
# reg_x.fit(feature_train, label_x)
# """ 训练z轴超参数 """
# reg_z.fit(feature_train, label_z)

# """ 将x轴超参数写入joblib储存 """
# gpr_x=reg_x.fit(feature_train, label_x)
# joblib.dump(gpr_x, 'D:/Program Files/Pycharm/Projects/基于强化学习的球杆系统人机协作控制/GPR/gpr_x.joblib')
# """ 将z轴超参数写入joblib储存 """
# gpr_z=reg_z.fit(feature_train, label_z)
# joblib.dump(gpr_z, 'D:/Program Files/Pycharm/Projects/基于强化学习的球杆系统人机协作控制/GPR/gpr_z.joblib')

""" 读取joblib中训练好的超参数 """
gpr_x = joblib.load('D:/Program Files/Pycharm/Projects/基于强化学习的球杆系统人机协作控制/GPR/gpr_x.joblib')
gpr_z = joblib.load('D:/Program Files/Pycharm/Projects/基于强化学习的球杆系统人机协作控制/GPR/gpr_z.joblib')

output_x, err_x  = gpr_x.predict(feature_test, return_std=True)
output_z, err_z  = gpr_z.predict(feature_test, return_std=True)
result_x = output_x.ravel()
result_z = output_z.ravel()
uncertainty_x = 1.96 * err_x
uncertainty_z = 1.96 * err_z

""" 添加字体 """
font1 = {'family': 'Times new Roman', 'size': 15}
font2 = {'family': 'FangSong', 'size': 15}

""" 输出图片 """
plt_x = np.arange(0, 100, 1).reshape(-1, 1)
plt_z = np.arange(0, 100, 1).reshape(-1, 1)

plt.figure(figsize=(10, 5))
plt.plot(plt_x, result_x)
plt.plot(plt_x, actual_x)
plt.fill_between(plt_x.flatten(), result_x + uncertainty_x, result_x - uncertainty_x, alpha=0.2)

plt.plot(np.arange(len(actual_x)), actual_x, linewidth=1.0, label='真实值', c='peru')
plt.plot(np.arange(len(result_x)), result_x, linewidth=2.0, label='预测值', c="dodgerblue")

plt.legend(loc='upper right', prop=font2, frameon=False)
plt.xlabel('step', fontdict=font1)
plt.ylabel('手部轨迹x轴速度/$m*s^{-1}$', fontdict=font2)

plt.show()

plt.figure(figsize=(10, 5))
plt.plot(plt_z, result_z)
plt.plot(plt_z, actual_z)
plt.fill_between(plt_z.flatten(), result_z + uncertainty_z, result_z - uncertainty_z, alpha=0.2)

plt.plot(np.arange(len(actual_z)), actual_z, linewidth=1.0, label='真实值', c='peru')
plt.plot(np.arange(len(result_z)), result_z, linewidth=2.0, label='预测值', c="dodgerblue")

plt.legend(loc='upper right', prop=font2, frameon=False)
plt.xlabel('step\n', fontdict=font1)
plt.ylabel('手部轨迹z轴速度/$m*s^{-1}$', fontdict=font2)

plt.show()
