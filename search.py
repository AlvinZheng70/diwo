import numpy as np

import generator
import main

seedData = generator.SeedData().get_seeds(50, 10)
jobs, LB = generator.generate_processing_times(seedData[0], 10, 50)

makespan = []
b_i = []
for x in range(100):
    pi = np.random.permutation(len(jobs))
    n=len(jobs)
    m=len(jobs[0])
    blocking_idle_time = np.zeros((n,m))
    d = np.zeros((n, m))
    for j in range(n):
        main.implement_one_line_of_d(d, jobs, j, pi[j])
    for i in range(1, n):
        for j in range(m):
            blocking_idle_time[i][j] = d[i][j] - jobs[pi[i]][j] - d[i - 1][j] + blocking_idle_time[i - 1][j]
    makespan.append(d[n-1][m-1])
    summ=0
    for j in range(m):
        summ+=blocking_idle_time[n-1][j]
    b_i.append(summ)

from scipy.stats import pearsonr

corr_AB, _ = pearsonr(makespan, b_i)
print(f"相关系数AB: {corr_AB}")
print(makespan)
print(b_i)