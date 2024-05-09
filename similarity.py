import numpy as np
def kendall_tau_distance(p, q):
    n = len(p)
    concordant_pairs = 0
    discordant_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (p[i] < p[j] and q[i] < q[j]) or (p[i] > p[j] and q[i] > q[j]):
                concordant_pairs += 1
            elif (p[i] < p[j] and q[i] > q[j]) or (p[i] > p[j] and q[i] < q[j]):
                discordant_pairs += 1

    return concordant_pairs - discordant_pairs


# 示例数据
p = [0, 1, 2, 3, 4, 5]
q = [5, 0, 1, 2, 3, 4]

# 计算 Kendall Tau 距离
distance = kendall_tau_distance(p, q)
print("Kendall Tau 距离:", distance)

def cosine_similarity(data1, data2):
    # 将数据转换为 NumPy 数组
    np_data1 = np.array(data1)
    np_data2 = np.array(data2)

    # 计算余弦相似度
    dot_product = np.dot(np_data1, np_data2)
    norm1 = np.linalg.norm(np_data1)
    norm2 = np.linalg.norm(np_data2)
    similarity = dot_product / (norm1 * norm2)

    return similarity

# 计算余弦相似度
similarity = cosine_similarity(p, q)
print("余弦相似度:", similarity)
