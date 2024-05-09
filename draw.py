import matplotlib.pyplot as plt
import numpy as np

import generator
import main


def draw_schedule(jobs, pi):
    m = len(jobs[0])
    n = len(jobs)
    blocking_time = 0
    idle_time = 0

    fig, ax = plt.subplots()
    d = np.zeros((n, m))

    for i in range(n):
        main.implement_one_line_of_d(d, jobs, i, pi[i])

    # 绘制阻塞部分和处理时间部分
    for i in range(n): # pi的第i个
        for j in range(m):
            x = d[i][j - 1] if j != 0 else (d[i - 1][0] if i != 0 else 0)
            processing_time = jobs[pi[i]][j]
            ax.barh(j, width=processing_time, left=x, height=1,
                    color='gray', alpha=0.3, edgecolor='black')
            if d[i][j] - x > processing_time:
                ax.barh(j, width=d[i][j] - x - processing_time, left=x + processing_time, height=1, color='red', alpha=0.5)
                blocking_time += d[i][j] - x - processing_time
            if i != 0 and x > d[i - 1][j]:
                ax.barh(j, width=x - d[i - 1][j], left=d[i - 1][j], height=1, color='blue', alpha=0.3)
                idle_time += x - d[i - 1][j]

    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Block Flow Shop Scheduling')

    # 设置工件标签
    ax.set_yticks(range(m))
    ax.set_yticklabels([f'Machine {i + 1}' for i in range(m)])

    ax.spines['bottom'].set_position(('data', -0.5))
    plt.text(main.calculate_cost(pi, jobs), -0.3,
             "blocking_time = " + str(int(blocking_time)) + "\nidle_time = " + str(int(idle_time)), fontsize=12,
             ha='right')
    plt.show()


seedData = generator.SeedData().get_seeds(20, 5)
jobs, LB = generator.generate_processing_times(seedData[0], 5, 20)
pi_best = [2, 16, 8, 13, 10, 5, 4, 17, 3, 9, 6, 11, 18, 14, 7, 15, 0, 1, 12, 19]
# print(jobs[pi_best,:])


# draw_schedule(jobs, pi_best)
