import numpy as np
from scipy.stats import pearsonr
import generator
import main

seedData = generator.SeedData().get_seeds(20, 5)
jobs, _ = generator.generate_processing_times(seedData[0], 5, 20)

pi_best = [2, 16, 8, 13, 10, 5, 4, 17, 3, 9, 6, 11, 18, 14, 7, 15, 0, 1, 12, 19]

jobs_var = np.var(jobs, axis=1)  # 行
reordered_jobs_var = jobs_var[pi_best]
correlation, _ = pearsonr(reordered_jobs_var, range(len(pi_best)))
print("工件处理时间方差和列位置的相关系数:", correlation)

jobs_sum = np.sum(jobs, axis=1)
reordered_jobs_sum = jobs_sum[pi_best]
correlation, _ = pearsonr(reordered_jobs_sum, range(len(pi_best)))
print("工件处理时间总和和列位置的相关系数:", correlation)

jobs_range = np.max(jobs, axis=1) - np.min(jobs, axis=1)
reordered_jobs_range = jobs_range[pi_best]
correlation, _ = pearsonr(reordered_jobs_range, range(len(pi_best)))
print("工件处理时间极值差和列位置的相关系数:", correlation)

# 相邻两组以机器为单位的和的方差
reordered_jobs = jobs[pi_best, :]
shifted_data = np.roll(reordered_jobs, shift=-1, axis=0)
merge_var = np.var((reordered_jobs + shifted_data)[:19, :], axis=1)
merge_var2 = np.var((jobs + np.roll(jobs, shift=-1, axis=0))[:19, :], axis=1)
print(("最优排列平均值:", np.average(merge_var)), ("普通排列平均值:", np.average(merge_var2)), "暂时无效")
# 可能是只取了相邻两个 忽视了累积的影响

n = len(jobs)
m = len(jobs[0])


# 排列欧氏距离、曼哈顿距离和切比雪夫距离
def distance(pi, jobs):
    num_jobs = len(jobs)
    num_machines = len(jobs[0])
    d = np.zeros((num_jobs, num_machines))
    for i in range(num_jobs):
        main.implement_one_line_of_d(d, jobs, i, pi[i])
    diff = (np.roll(d, shift=-1, axis=1) - d)[:, :num_machines - 1]
    # 排列的上下两块的差值和下一个工件的时间的距离
    euclidean_distance = []
    manhattan_distance = []
    chebyshev_distance = []
    for i in range(num_machines - 1):
        array1 = diff[i]
        array2 = jobs[pi[i + 1]][:num_machines - 1]
        # print(array1)
        # print(array2)
        euclidean_distance.append(np.linalg.norm(array1 - array2))
        manhattan_distance.append(np.sum(np.abs(array1 - array2)))
        chebyshev_distance.append(np.max(np.abs(array1 - array2)))
    return np.average(euclidean_distance), np.average(manhattan_distance), np.average(chebyshev_distance)


# 随机生成十个个体，其中一个选择最优个体，计算距离的相关性
POP = [pi_best]
while len(POP) < 10:
    pi = np.random.permutation(n)
    if all((pi != existing_pi).any() for existing_pi in POP):
        POP.append(pi)
cost_list = []
euclidean_distance_list = []
manhattan_distance_list = []
chebyshev_distance_list = []
for i in range(len(POP)):
    e_, m_, c_ = distance(POP[i], jobs)
    euclidean_distance_list.append(e_)
    manhattan_distance_list.append(m_)
    chebyshev_distance_list.append(c_)
    cost_list.append(main.calculate_cost(POP[i],jobs))

# 距离主要考虑下一个工件最好填补差值
print("欧氏距离和makespan的相关系数:", pearsonr(euclidean_distance_list, cost_list)[0])
print("曼哈顿距离和makespan的相关系数:", pearsonr(manhattan_distance_list, cost_list)[0])
print("切比雪夫距离和makespan的相关系数:", pearsonr(chebyshev_distance_list, cost_list)[0],"PS: 两个点之间各个坐标数值差的绝对值的最大值。")
print("PF启发式算法考虑的是阻塞时间和空闲时间，此处考虑的是匹配程度")

# 前述——和的方差——不合理


