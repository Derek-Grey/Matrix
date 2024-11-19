import numpy as np

# 设置随机种子以获得可重复的结果
np.random.seed(0)

# 生成一个3维的随机向量v
v = np.random.randn(3, 1)
# 归一化v得到单位向量u
u = v / np.linalg.norm(v)

# 生成一个3维的矩阵V，包含30个随机向量
V = np.random.randn(3, 30)
# 归一化V的每一列得到单位向量Uj
U = V / np.linalg.norm(V, axis=0)

# 计算u和每个Uj的点积的绝对值
dot_products = np.abs(np.dot(u.T, U))

# 计算这些点积绝对值的平均值
average_dot_product = np.mean(dot_products)

# 打印结果
print("平均点积的绝对值:", average_dot_product)

# 根据微积分计算的理论平均值
theoretical_average = 2 / np.pi
print("理论平均值:", theoretical_average)