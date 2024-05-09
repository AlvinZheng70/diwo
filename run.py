import multiprocessing
import time
from datetime import datetime

import main
import generator
import sqlite3


def data_handle(params, permutation, makespan, version):
    # 存储结果
    conn = sqlite3.connect('data.db')  # 连接到 SQLite 数据库（如果不存在则会自动创建）
    cursor = conn.cursor()  # 创建一个游标对象，用于执行 SQL 语句
    # 创建一个表
    cursor.execute('''CREATE TABLE IF NOT EXISTS results (
                            n INTEGER,
                            m INTEGER,
                            seed INTEGER,
                            Pmax INTEGER,
                            Smin INTEGER,
                            Smax INTEGER,
                            sigma_min INTEGER,
                            sigma_max INTEGER,
                            pls FLOAT,
                            lambd INTEGER,
                            x INTEGER,
                            tmax INTEGER,
                            permutation TEXT NOT NULL,
                            makespan INTEGER,
                            version TEXT NOT NULL,
                            time TEXT NOT NULL
                        )''')
    # 插入数据
    cursor.execute(
        "INSERT INTO results (n, m, seed, Pmax, Smin, Smax, sigma_min, sigma_max, pls, lambd, x, tmax, permutation, makespan, version, time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (int(params['tuple'][0]), int(params['tuple'][1]), int(params['tuple'][2]), int(params['Pmax']),
         int(params['Smin']), int(params['Smax']),
         int(params['sigma_min']), int(params['sigma_max']), float(params['pls']), int(params['lambd']),
         int(params['x']), int(params['tmax']),
         ', '.join(str(x) for x in permutation), int(makespan), version,
         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
    # 提交事务
    conn.commit()
    # 关闭游标和连接
    cursor.close()
    conn.close()


def evolution_algorithm(params):
    Pmax = params['Pmax']
    Smin = params['Smin']
    Smax = params['Smax']
    sigma_min = params['sigma_min']
    sigma_max = params['sigma_max']
    pls = params['pls']
    n, m, seed = params['tuple']
    lambd = params['lambd']
    x = params['x']
    tmax = params['tmax']
    cost_function = params['cost_function']
    jobs, LB = generator.generate_processing_times(seed, m, n)
    runs_per_instance = 1
    for i in range(runs_per_instance):
        result = main.DIWO(Pmax, Smin, Smax, sigma_min, sigma_max, pls, jobs, lambd, x, tmax, cost_function)
        data_handle(params, result, main.calculate_cost(result, jobs), version='局部搜索更改')
        print(n, '\t', m, '\t', seed, '\t', result, '\t', main.calculate_cost(result, jobs))


def safe_evolution_algorithm(params):
    try:
        return evolution_algorithm(params)
    except Exception as e:
        print("算法发生异常:", e)
        return None


if __name__ == '__main__':
    n_values = [20, 50, 100, 200, 500]  # n 的所有可能取值
    m_values = [5, 10, 20]  # m 的所有可能取值
    jobs_tuples = []
    seedData = generator.SeedData()
    for n in n_values:
        for m in m_values:
            seeds = seedData.get_seeds(n, m)
            if seeds is not None:
                jobs_tuples.append((n, m, seeds[0]))
            '''if seeds is not None:
                jobs_tuples.extend((n, m, seed) for seed in seeds)'''
    # 参数列表
    params_list = [{'Pmax': 10, 'Smin': 0, 'Smax': 7, 'sigma_min': 0, 'sigma_max': 5,
                    'pls': 0.15, 'tuple': jobs_tuple, 'lambd': 10, 'x': 5, 'tmax': 30 * 60,
                    'cost_function': main.calculate_cost}
                   for jobs_tuple in jobs_tuples]

    # 创建一个进程池，使用所有可用的核心
    pool = multiprocessing.Pool()

    # 使用 map 函数并行运行进化算法
    results = pool.map(safe_evolution_algorithm, params_list)

    # 关闭进程池
    pool.close()
    pool.join()

    # 处理结果
