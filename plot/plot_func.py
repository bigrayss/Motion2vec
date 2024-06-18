import numpy as np
import matplotlib.pyplot as plt

# Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# tanh 函数
def tanh(x):
    return np.tanh(x)

# ReLU 函数
def relu(x):
    return np.maximum(0, x)

# GeLU 函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# 生成输入值
x = np.linspace(-5, 5, 100)

# 计算各函数的输出值
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_gelu = gelu(x)

# 绘制函数图像
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(x, y_sigmoid, label='Sigmoid', color='blue')
axs[0, 0].set_title('Sigmoid')
axs[0, 0].grid(True)

axs[0, 1].plot(x, y_tanh, label='tanh', color='red')
axs[0, 1].set_title('tanh')
axs[0, 1].grid(True)

axs[1, 0].plot(x, y_relu, label='ReLU', color='green')
axs[1, 0].set_title('ReLU')
axs[1, 0].grid(True)

axs[1, 1].plot(x, y_gelu, label='GeLU', color='purple')
axs[1, 1].set_title('GeLU')
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('activation_functions.png')
plt.show()