import matplotlib.pyplot as plt

# 数据
data1 = [5.7802474e-05, 2.4290968e-05, 2.078478e-05, 1.2583401e-05, 8.087037e-06, 7.423926e-06, 6.6212915e-06, 5.959177e-06, 5.47233e-06, 5.097243e-06, 4.781258e-06, 4.499277e-06, 4.301676e-06, 3.8659327e-06, 3.6289134e-06]
data2 = [5.8310372e-05, 2.3682676e-05, 1.9542178e-05, 1.0718356e-05, 8.202693e-06, 7.4766954e-06, 6.8599725e-06, 6.1761257e-06, 5.5625896e-06, 5.18275e-06, 4.6848495e-06, 4.350807e-06, 4.00769e-06, 3.704654e-06, 3.425212e-06]
data3 = [6.0534134e-05, 2.4848892e-05, 1.3189259e-05, 1.0688319e-05, 8.7948165e-06, 8.124485e-06, 7.008498e-06, 6.2055406e-06, 5.632478e-06, 5.0967765e-06, 4.889001e-06, 4.732309e-06, 4.2534666e-06, 3.873733e-06, 3.5445867e-06]
data4 = [0.00010441831, 5.828433e-05, 3.8782033e-05, 2.94264e-05, 2.5275243e-05, 1.780754e-05, 1.446156e-05, 1.3349202e-05, 1.2051925e-05, 1.1807393e-05, 1.1086314e-05, 1.0655439e-05, 1.0158472e-05, 9.679657e-06, 9.6313115e-06]
data5 = [6.130018e-05, 2.3366747e-05, 1.4033707e-05, 1.0475946e-05, 8.729982e-06, 8.03889e-06, 7.363971e-06, 6.651908e-06, 6.097369e-06, 5.6385584e-06, 5.23298e-06, 4.9745636e-06, 4.5774277e-06, 4.2421293e-06, 4.2178654e-06]
data6 = [7.480381e-05, 2.9534514e-05, 2.436702e-05, 1.9828845e-05, 1.4758623e-05, 1.0130296e-05, 9.211164e-06, 8.1894905e-06, 7.557181e-06, 7.075214e-06, 6.4318633e-06, 6.0425123e-06, 5.7357224e-06, 5.351757e-06, 4.92719e-06]

# 横坐标
x = list(range(1, len(data1) + 1))

fig, axs = plt.subplots(1, 3, figsize=(12, 4))


axs[0].plot(x, data1, label='P=0.6')
axs[0].plot(x, data2, label='P=0.7')
axs[0].plot(x, data3, label='P=0.8')
axs[0].set_title('Ablation experiments for P')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend(loc='upper right')

axs[1].plot(x, data2, label='M=3')
axs[1].plot(x, data4, label='M=1')
axs[1].set_title('Ablation experiments for M')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend(loc='upper right')

axs[2].plot(x, data2, label='K=8')
axs[2].plot(x, data5, label='K=4')
axs[2].plot(x, data6, label='k=6')
axs[2].set_title('Ablation experiments for K')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Loss')
axs[2].legend(loc='upper right')


plt.tight_layout()
# 保存图片
plt.savefig('pretrain.png')
# 显示图形
plt.show()