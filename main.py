import sys
import numpy as np
import random
import time

from matplotlib import pyplot as plt

import generator


def initialize_population(N0, lambd, x, jobs):
    # 条件检查
    if lambd > len(jobs) or lambd < 0:
        raise ValueError("Argument lambd is illegal!")
    if x > len(jobs) or x < 0:
        raise ValueError("Argument x is illegal!")

    POP = []
    pi1 = np.argsort(np.sum(jobs, axis=1))  # 按总处理时间非递减的顺序排序
    k = 0

    while k < x:
        pi = np.zeros_like(pi1)
        pi[0] = pi1[k]
        U = set(range(len(jobs)))
        U.remove(pi1[k])
        i = 1
        d = np.zeros((len(jobs), len(jobs[0])))
        while U:
            # 循环内确定位置i，首先完善位置i-1的离开时间
            implement_one_line_of_d(d, jobs, i - 1, pi[i - 1])
            # 测试U中的各个工件
            min_idle_blocking_time = float('inf')
            min_idx = None
            for j in U:
                implement_one_line_of_d(d, jobs, i, j)
                sum_idle_blocking = np.sum(d[i] - d[i - 1] - jobs[j])
                if sum_idle_blocking < min_idle_blocking_time:
                    min_idle_blocking_time = sum_idle_blocking
                    min_idx = j
            U.remove(min_idx)
            pi[i] = min_idx
            i += 1
        # 插入评估
        pi2 = pi[:len(pi) - lambd].copy()
        for q in range(len(pi) - lambd, len(pi)):
            best_pos, cmax = find_the_best_pos(pi2, pi[q], jobs)
            pi2 = np.insert(pi2, best_pos, pi[q])
        if all((pi2 != existing_pi).any() for existing_pi in POP):
            POP.append(pi2)
        k += 1
    POP = sorted(POP, key=lambda y: calculate_cost(y, jobs))  # 根据完成时间对任务进行排序

    while len(POP) < N0:
        pi = np.random.permutation(len(jobs))
        if all((pi != existing_pi).any() for existing_pi in POP):
            POP.append(pi)
    return POP


def find_the_best_pos(seq, to_be_inserted, jobs):
    m = len(jobs[0])
    n = len(seq)  # 待插入序列的长度

    # 计算待插入序列的离开时间矩阵和尾部时间矩阵
    d = np.zeros((n, m))
    f = np.zeros((n, m))
    for i in range(n):
        implement_one_line_of_d(d, jobs, i, seq[i])
        implement_one_line_of_f(f, jobs, n - i - 1, seq[n - i - 1])

    d0 = np.zeros(m)
    cmax = []
    for pos in range(n + 1):
        if pos == 0:
            for j in range(m):
                d0[j] = jobs[to_be_inserted][j] + (d0[j - 1] if j != 0 else 0)
        else:
            for j in range(m):
                d0[j] = max(jobs[to_be_inserted][j] + (d0[j - 1] if j != 0 else d[pos - 1][0]),
                            d[pos - 1][j + 1] if j != len(jobs[0]) - 1 else 0)
        if pos != n:
            cmax.append(np.max(d0 + f[pos]))
        else:
            cmax.append(d0[m - 1])
    return cmax.index(min(cmax)), min(cmax)


def implement_one_line_of_d(d, jobs, i, implemented):
    if i == 0:
        for j in range(len(jobs[0])):
            d[0][j] = jobs[implemented][j] + (d[0][j - 1] if j != 0 else 0)
    else:
        for j in range(len(jobs[0])):
            # max(上一个机器的离开时间+这一个机器的持续时间，上一个工件在下一个机器的离开时间)
            d[i][j] = max(jobs[implemented][j] + (d[i][j - 1] if j != 0 else d[i - 1][0]),
                          d[i - 1][j + 1] if j != len(jobs[0]) - 1 else 0)


def implement_one_line_of_f(f, jobs, i, implemented):
    m = len(jobs[0])
    if i == len(f) - 1:
        for j in range(m - 1, -1, -1):
            f[i][j] = jobs[implemented][j] + (f[i][j + 1] if j != m - 1 else 0)
    else:
        for j in range(m - 1, -1, -1):
            f[i][j] = max(jobs[implemented][j] + (f[i][j + 1] if j != m - 1 else f[i + 1][j]),
                          f[i + 1][j - 1] if j != 0 else 0)


def calculate_fitness(pi_i, pi_worst, pi_best, cost_function, S_max, S_min, jobs, epsilon=sys.float_info.epsilon):
    """
    计算个体的适应度。

    参数:
    pi_i (array-like): 当前个体的排列。
    pi_worst (array-like): 最差个体的排列。
    pi_best (array-like): 最好个体的排列。
    cost_function (function): 计算排列成本的函数。
    S_max (float): 适应度的最大值。
    S_min (float): 适应度的最小值。
    jobs (array-like): 工件在各个机器上的的执行时间
    epsilon (float): 微小值，避免除以零。

    返回值:
    float: 个体的适应度。
    """
    # 计算当前个体的排列成本
    cost_pi_i = cost_function(pi_i, jobs)

    # 计算最差个体的排列成本和最好个体的排列成本
    cost_pi_worst = cost_function(pi_worst, jobs)
    cost_pi_best = cost_function(pi_best, jobs)

    # 计算适应度
    fitness = np.floor(
        (cost_pi_worst - cost_pi_i + epsilon) / (cost_pi_worst - cost_pi_best + epsilon) * (S_max - S_min) + S_min)
    return int(fitness)


def calculate_cost(pi, jobs):
    """
    计算排列的成本
    :param pi: 排列
    :param jobs: 工件的加工时间信息
    :return: 排列的总加工时间
    """
    d = np.zeros((len(pi), len(jobs[0])))
    for i in range(len(pi)):
        implement_one_line_of_d(d, jobs, i, pi[i])
    return d[len(pi) - 1][len(jobs[0]) - 1]


def compute_sigma_i_k(pi_i, pi_worst, pi_median, cost_function, sigma_max, sigma_min,
                      t, t0, tmax, jobs, epsilon=sys.float_info.epsilon):
    """
    计算某个时间点某个个体的对应标准差
    :param pi_i:
    :param pi_worst:
    :param pi_median:
    :param cost_function:
    :param sigma_max: sigma_k的最大值
    :param sigma_min: sigma_k的最小值
    :param t:
    :param t0: 算法开始时间
    :param tmax: 最大的CPU执行时间
    :param jobs:
    :param epsilon:
    :return: 某个时间点某个个体的对应标准差
    """
    # 首先根据执行时间决定初步标准差
    sigma_k = abs((1 - (t - t0) / tmax) * (sigma_max - sigma_min) + sigma_min)

    # 计算排列的最大完成时间
    cost_pi_i = cost_function(pi_i, jobs)
    cost_pi_median = cost_function(pi_median, jobs)

    # 根据杂草的表现最终决定标准差
    if cost_pi_i < cost_pi_median:  # pi_i最大完成时间较小，表现较好
        return sigma_k
    else:
        cost_pi_worst = cost_function(pi_worst, jobs)
        return sigma_k * ((cost_pi_i - cost_pi_median) / (cost_pi_worst - cost_pi_median + epsilon) * 0.5 + 1)


def generate_di(sigma_i_k, sigma_max, sigma_min, jobs):
    """
    生成随机扩散时移除的序列长度
    :param sigma_i_k: 该个体在某个时间点对应的标准差
    :param sigma_max:
    :param sigma_min:
    :param jobs:
    :return: 移除的序列长度
    """
    # 根据标准差生成长度
    di = int(np.floor(abs(np.random.normal(0, sigma_i_k))))

    # 避免空间扩散简化为 NEH
    if di > len(jobs) / 2 or di < sigma_min:
        di = np.floor(sigma_min + (sigma_max - sigma_min) * np.random.rand())
    return int(di)


def random_insertion_space_spread(POP, S_min, S_max, sigma_min, sigma_max, pi_best, pi_worst, pi_median, t0, tmax, jobs,
                                  cost_function):
    """
    基于随机插入的空间扩散算法
    :param jobs:
    :param POP:
    :param S_min: 种子的最小数量
    :param S_max: 种子的最大数量
    :param sigma_min:
    :param sigma_max:
    :param pi_best:
    :param pi_worst:
    :param pi_median:
    :param t0: 算法的开始时间
    :param tmax: 最大的CPU执行时间
    :param cost_function:
    :return: 空间扩散后的种群
    """
    POP_prime = []  # 输出种群
    for pi in POP:
        S_i = calculate_fitness(pi, pi_worst, pi_best, cost_function, S_max, S_min, jobs)  # 计算适应度
        # 获取当前CPU时间
        t = time.time()
        sigma_i_k = compute_sigma_i_k(pi, pi_worst, pi_median, cost_function, sigma_max, sigma_min, t, t0, tmax,
                                      jobs)
        for j in range(S_i):
            di = generate_di(sigma_i_k, sigma_max, sigma_min, jobs)
            random_indices = np.random.choice(len(pi), di, replace=False)
            sorted_indices = np.sort(random_indices)
            pi_R = pi[sorted_indices]  # 注： 切片操作返回副本
            pi_prime = np.delete(pi, sorted_indices)
            pi_R = sorted(pi_R, key=lambda x: sum(jobs[x]))
            for k in range(di):
                best_pos, cmax = find_the_best_pos(pi_prime, pi_R[k], jobs)
                pi_prime = np.insert(pi_prime, best_pos, pi_R[k])
            POP_prime.append(pi_prime)
    return POP_prime


def local_search(pi, pi_r, jobs):
    pi_r = pi_r.copy()
    n = len(pi)  # 排列长度，也即总工件数
    cntr = 0
    j = -1
    while cntr < n:
        j = (j + 1) % n
        index_to_remove = np.where(pi == pi_r[j])[0]
        pi_prime = np.delete(pi, index_to_remove)
        best_pos, cmax = find_the_best_pos(pi_prime, pi_r[j], jobs)
        if calculate_cost(pi, jobs) > cmax:
            pi = np.insert(pi_prime, best_pos, pi_r[j])
            cntr = 0
        else:
            cntr += 1
    pi_temp = pi.copy()

    max_jobs = random.sample(range(0, n), int(n / 10))

    mask = np.isin(pi, max_jobs)
    pi_left = pi[~mask]

    for element in max_jobs:
        index_to_insert = np.random.randint(0, len(pi_left) + 1)  # 随机选择插入位置
        pi_left = np.insert(pi_left, index_to_insert, element)

    pi = pi_left
    pi_r = pi_r.copy()
    n = len(pi)  # 排列长度，也即总工件数

    cntr = 0
    j = -1
    while cntr < n:
        j = (j + 1) % n
        # print(n,j)
        index_to_remove = np.where(pi == pi_r[j])[0]
        pi_prime = np.delete(pi, index_to_remove)
        # print(pi_prime,pi_r[j])
        best_pos, cmax = find_the_best_pos(pi_prime, pi_r[j], jobs)
        if calculate_cost(pi, jobs) > cmax:
            pi = np.insert(pi_prime, best_pos, pi_r[j])
            cntr = 0
        else:
            cntr += 1

    if (calculate_cost(pi, jobs) > calculate_cost(pi_temp, jobs)):
        return pi_temp;
    else:
        return pi


def shuffle_local_search(pi, pi_r, jobs):
    """
    参考局部搜索
    :param pi: 搜索起点
    :param pi_r: 参考序列
    :param jobs:
    :return: 搜索后序列
    """
    pi_r = pi_r.copy()
    n = len(pi)  # 排列长度，也即总工件数
    cntr = 0
    j = -1
    flag = 0

    while flag != 2:
        while cntr < n:
            j = (j + 1) % n
            index_to_remove = np.where(pi == pi_r[j])[0]
            pi_prime = np.delete(pi, index_to_remove)
            best_pos, cmax = find_the_best_pos(pi_prime, pi_r[j], jobs)
            if calculate_cost(pi, jobs) > cmax:
                pi = np.insert(pi_prime, best_pos, pi_r[j])
                cntr = 0
            else:
                cntr += 1
        if flag == 0:
            fisher_yates_shuffle(pi_r)
        flag += 1

    return pi


def fisher_yates_shuffle(pi):
    """
    Fisher-Yates shuffle算法，用于随机重排一个排列
    :param pi:
    :return: 随机重排后的排列
    """
    n = len(pi)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        pi[i], pi[j] = pi[j], pi[i]


def competition_exclusion(POP, POP_prime, P_max, jobs):
    """
    竞争性排除
    :param POP:
    :param POP_prime:
    :param P_max: 种群最大植株数量
    :param jobs:
    :return: 竞争后种群
    """
    POP_double_prime = []
    POP_double_prime.extend(POP)
    POP_double_prime.extend(POP_prime)
    sorted_POP_double_prime = sorted(POP_double_prime, key=lambda x: calculate_cost(x, jobs))  # 根据完成时间对任务进行排序

    POP_result = []
    j = 1
    POP_result.append(sorted_POP_double_prime[0])
    while len(POP_result) < P_max and j < len(sorted_POP_double_prime):
        flag = True
        for pi in POP_result:
            if distance(pi, sorted_POP_double_prime[j]) == 0:
                flag = False
                break
        if flag:
            POP_result.append(sorted_POP_double_prime[j])
            j += 1
        else:
            j += 1
    return POP_result

def distance(pi1, pi2):
    """
    检查两个个体的相似性
    :param pi1:
    :param pi2:
    :return: 两个个体相似性的量化指标
    """
    return sum(1 for i in range(len(pi1)) if pi1[i] != pi2[i])


def DIWO(Pmax, Smin, Smax, sigma_min, sigma_max, pls, jobs, lambd, x, tmax, cost_function):
    POP = initialize_population(Pmax, lambd, x, jobs)
    k = 1
    t0 = time.time()
    POP = sorted(POP, key=lambda x: cost_function(x, jobs))
    while True:
        # 空间扩散
        t_zero = time.time()
        POP_prime = random_insertion_space_spread(POP, Smin, Smax, sigma_min, sigma_max, POP[0],
                                                  POP[len(POP) - 1],
                                                  POP[int(len(POP) / 2)], t0, tmax, jobs, cost_function)
        t_one = time.time()
        # 指向同一个对象
        for i in range(len(POP_prime)):
            if random.random() < pls:
                pi_prime = local_search(POP_prime[i], POP[0], jobs)  # 局部搜索过程
                POP_prime[i] = pi_prime
        t_two = time.time()
        POP = competition_exclusion(POP, POP_prime, Pmax, jobs)
        t_three = time.time()
        k += 1
        # print(t_one-t_zero,t_two-t_one,t_three-t_two)
        # 判断是否满足终止准则
        if time.time() - t0 >= tmax:
            break
    return POP[0]  # 返回最佳个体


seedData = generator.SeedData().get_seeds(20, 5)
jobs, LB = generator.generate_processing_times(seedData[0], 5, 20)

# print(calculate_cost([ 2, 16, 15,  5, 13, 19, 11, 10,  4,  9,  6,  7,  8, 14, 12,  0, 18, 3,  1, 17], jobs))
# 0 <= sigma_min <= sigma_max <= len(jobs)
result = DIWO(Pmax=10,
              Smin=0,
              Smax=7,
              sigma_min=5,
              sigma_max=10,
              pls=0.15,
              jobs=jobs,
              lambd=10,
              x=5,
              tmax=30*60,
              cost_function=calculate_cost)
print(result)
print(calculate_cost(result, jobs))
