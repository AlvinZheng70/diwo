import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import generator
from itertools import permutations

# 假设你有一个排列的列表
# 例如：permutations = [[1, 2, 3], [3, 1, 2], ...]
import main

jobs, LB = generator.generate_processing_times(379008056, 3 ,7)
print(jobs)
jobs = [[26, 63, 37, 62, 54],
 [38, 23, 54, 44,  9],
 [27, 45, 35, 10, 30],
 [88, 86, 59, 23, 31],
 [95, 43, 43, 64, 92],
 [55, 43, 50, 47,  7],
 [54, 40, 59, 68, 14]]
# 生成10以内的所有数的排列
numbers = list(range(7))
permutations = list(permutations(numbers))
objective_values=[]
for x in permutations:
    objective_values.append(main.calculate_cost(x,jobs))


# 计算Kendall tau距离矩阵
def kendall_tau_distance(perm1, perm2):
    n = len(perm1)
    num_discordant_pairs = sum((perm1[i] < perm1[j]) != (perm2[i] < perm2[j]) for i in range(n) for j in range(i+1, n))
    return num_discordant_pairs

# cos
from numpy.linalg import norm
def func(vector_a,vector_b):
    dot_product = np.dot(vector_a, vector_b)
    # 计算向量的模长
    norm_a = norm(vector_a)
    norm_b = norm(vector_b)
    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_a * norm_b)
    return cosine_similarity

import Levenshtein

# 计算 Levenshtein 编辑距离
def leven(pi1,pi2):
    return Levenshtein.distance(pi1, pi2)

print(leven([0,1,2,3,4],[1,2,3,4,0]))
print(leven([0,1,2,3,4],[2,3,4,1,0]))
dist_matrix = squareform(pdist(permutations, metric=leven))

# 使用MDS进行降维
# mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
# reduced_coords = mds.fit_transform(dist_matrix)

# 或者使用t-SNE进行降维
tsne = TSNE(n_components=2, metric='precomputed', random_state=42)
reduced_coords = tsne.fit_transform(dist_matrix)

# 绘图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_coords[:, 0], reduced_coords[:, 1], c=objective_values, cmap='viridis', s=50)
plt.colorbar(scatter, label='Objective Function Value')
plt.title('Solution Space Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()