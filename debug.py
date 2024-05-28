import generator
import numpy as np

import main_memo


def beam_search_constructive_heuristic(jobs, x, a):
    # Step 1: Initial Order
    m = len(jobs[0])
    n = len(jobs)
    # Calculate xi_j for all jobs
    xi = np.zeros(n)
    for i in range(n):
        # Calculate weighted idle time w_j
        w_j = sum(np.sum(jobs[i, :j]) / j for j in range(1, m)) * m
        # Calculate xi_j
        xi[i] = (n - 2) / 4 * w_j + np.sum(jobs[i, :])
    # Get initial order alpha
    alpha = np.argsort(xi)
    S = {l: [alpha[l]] for l in range(x)}  # selected nodes集合
    U = {l: {job for job in range(n) if job not in S[l]} for l in range(x)}  # 记录结点对应的未被选择的工件
    F = {l: 0 for l in range(x)}  # 结点的F index记录
    departure_time = {}
    for l in range(x):
        departure_time[l] = np.zeros(m)
        for j in range(m):
            departure_time[l][j] = jobs[alpha[l]][j] + (departure_time[l][j - 1] if j != 0 else 0)
    delta_idle_time = {l: 0 for l in range(x)}
    delta_blocking_time = {l: 0 for l in range(x)}

    # Steps 2-5: Iterate over each job position
    candidates_blocking_time = {}
    candidates_idle_time = {}
    candidates_departure_time = {}
    for k in range(1, n - 1):  # k也表征selected nodes序列长度
        # Step 2-3: Candidate Nodes Creation and Evaluation
        G = {}
        for l in range(x):
            for v in U[l]:
                # calculate the sequence (l + v)
                d_l_v = np.zeros(m)
                for j in range(m):
                    d_l_v[j] = max(jobs[v][j] + (d_l_v[j - 1] if j != 0 else departure_time[l][0]),
                                   departure_time[l][j + 1] if j != m - 1 else 0)
                candidates_departure_time[(l, v)] = d_l_v
                e_l_v = np.roll(d_l_v, 1)
                e_l_v[0] = departure_time[l][0]
                candidates_blocking_time[(l, v)] = sum((d_l_v - jobs[v] - e_l_v)[:m - 1])
                candidates_idle_time[(l, v)] = sum(max(0, x) for x in (np.roll(d_l_v, 1) - departure_time[l])[1:])
                # print(n,m,F[l],candidates_departure_time[(l, v)][m - 1])
                G[(l, v)] = F[l] + a * (
                        candidates_idle_time[(l, v)] + candidates_blocking_time[(l, v)]) + \
                            candidates_departure_time[(l, v)][m - 1]
                # print(F[l], candidates_idle_time[(l, v)], candidates_blocking_time[(l, v)],candidates_departure_time[(l, v)][m - 1])

        # Step 4: Candidate Nodes Selection
        best_candidates = np.array([k for k, v in sorted(G.items(), key=lambda item: item[1])])[:x]

        # Step 5: Forecasting Phase
        new_delta_idle_time = {l: 0 for l in range(x)}
        new_delta_blocking_time = {l: 0 for l in range(x)}
        new_S = {}
        for idx, (l, q) in enumerate(best_candidates):
            new_S[idx] = S[l][:]
            new_S[idx].append(q)
            new_delta_blocking_time[idx] = delta_blocking_time[l] + candidates_blocking_time[(l, q)] * (n - k - 2) / n
            new_delta_idle_time[idx] = delta_idle_time[l] + candidates_idle_time[(l, q)] * (n - k - 2) / n
            set_jobs = U[l].copy()
            set_jobs.remove(q)
            artificial_job = np.mean(jobs[list(set_jobs), :], axis=0)
            d_artificial_job = np.zeros(m)
            for j in range(m):
                d_artificial_job[j] = max(
                    artificial_job[j] + (d_artificial_job[j - 1] if j != 0 else candidates_departure_time[(l, q)][0]),
                    candidates_departure_time[(l, q)][j + 1] if j != m - 1 else 0)
            new_delta_departure_time = candidates_departure_time[(l, q)][m - 1] + d_artificial_job[m - 1]
            departure_time[idx] = candidates_departure_time[(l, q)]
            F[idx] = new_delta_departure_time + a * (new_delta_blocking_time[idx] + new_delta_idle_time[idx])
        for l in range(x):
            S[l] = new_S[l]
            U[l] = {job for job in range(n) if job not in S[l]}
            delta_idle_time[l] = new_delta_idle_time[l]
            delta_blocking_time[l] = new_delta_blocking_time[l]
    for l in range(x):
        S[l].append(list(U[l])[0])
    a = []
    for i in range(x):
        a.append(main_memo.calculate_cost(S[i], jobs))
    POP = sorted(a)  # 根据完成时间对任务进行排序
    # print(POP)
    print("beem",np.average(POP[:5]))


# 数据集生成






seedData = generator.SeedData().get_seeds(200, 20)
jobs, LB = generator.generate_processing_times(seedData[0], 20, 200)

beam_search_constructive_heuristic(jobs, 5, 5)