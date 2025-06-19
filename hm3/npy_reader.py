import numpy as np

# 替换为你的文件路径
file_path = 'theta.npy'

# 读取 .npy 文件
data = np.load(file_path)

# 打印内容
print("数据形状:", data.shape)
print("数据类型:", data.dtype)
print("前几行数据:")
print(data[:5])  # 只打印前5行