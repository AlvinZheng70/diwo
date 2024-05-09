import numpy as np

import draw
import generator
import main
from main import find_the_best_pos

seedData = generator.SeedData().get_seeds(20, 5)
jobs, _ = generator.generate_processing_times(seedData[0], 5, 20)
pi_best = [2, 16, 8, 13, 10, 5, 4, 17, 3, 9, 6, 11, 18, 14, 7, 15, 0, 1, 12, 19]
m = len(jobs[0])
n = len(jobs)
pi = np.random.permutation(n)
print(pi, main.calculate_cost(pi,jobs))
# draw.draw_schedule(jobs,pi)

di = 4

# method 1：原方法
random_indices = np.random.choice(len(pi), di, replace=False)
sorted_indices = np.sort(random_indices)
pi_R = pi[sorted_indices]  # 注： 切片操作返回副本
pi_prime = np.delete(pi, sorted_indices)
pi_R = sorted(pi_R, key=lambda x: sum(jobs[x]))
for k in range(di):
    best_pos, cmax = find_the_best_pos(pi_prime, pi_R[k], jobs)
    pi_prime = np.insert(pi_prime, best_pos, pi_R[k])
print('1.0',pi_prime, '\t', main.calculate_cost(pi_prime, jobs))
# draw.draw_schedule(jobs,pi_prime)

# method 1.1：原方法
random_indices = np.random.choice(len(pi), di, replace=False)
sorted_indices = np.sort(random_indices)
pi_R = pi[sorted_indices]  # 注： 切片操作返回副本
pi_prime = np.delete(pi, sorted_indices)
pi_R = sorted(pi_R, key=lambda x: -sum(jobs[x]))
for k in range(di):
    best_pos, cmax = find_the_best_pos(pi_prime, pi_R[k], jobs)
    pi_prime = np.insert(pi_prime, best_pos, pi_R[k])
print('1.1', pi_prime, '\t', main.calculate_cost(pi_prime, jobs))
# draw.draw_schedule(jobs,pi_prime)

# method 2：匹配差值
pi_prime = np.delete(pi, sorted_indices)
d = np.zeros((n, m))
for k in range(di):
    for i in range(len(pi_prime)):
        main.implement_one_line_of_d(d, jobs, i, pi_prime[i])
    diff = (np.roll(d, shift=-1, axis=1) - d)
    for row in diff:
        row[-1] = 0
    chebyshev_distance = []
    for i in range(len(pi_prime)):
        chebyshev_distance.append(np.max(np.abs(diff[i] - jobs[pi_R[k]])))
    flag = False
    dis_min = 0
    dis_min_pos = 0
    for i in range(len(pi_prime) - 1):
        dis_min = min(chebyshev_distance)
        dis_min_pos = chebyshev_distance.index(dis_min)
        if dis_min < np.max(np.abs(diff[i] - jobs[pi_prime[i + 1]])):
            flag = True
            break
    if flag:
        pi_prime = np.insert(pi_prime, dis_min_pos + 1, pi_R[k])
    else:
        if main.calculate_cost(np.insert(pi_prime, 0, pi_R[k]), jobs) > main.calculate_cost(
                np.insert(pi_prime, len(pi_prime), pi_R[k]), jobs):
            pi_prime = np.insert(pi_prime, len(pi_prime), pi_R[k])
        else:
            pi_prime = np.insert(pi_prime, 0, pi_R[k])
print(pi_prime, '\t', main.calculate_cost(pi_prime, jobs))
# draw.draw_schedule(jobs,pi_prime)

# 表现较差，可能的原因是理想化了，指标太泛了
