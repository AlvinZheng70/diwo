import multiprocessing
import main
import generator


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
    ans = params['ans']
    jobs, LB = generator.generate_processing_times(seed, m, n)
    result = main.DIWO(Pmax, Smin, Smax, sigma_min, sigma_max, pls, jobs, lambd, x, tmax, cost_function)
    ans.extend((n, m, seed, result, main.calculate_cost(result, jobs)))
    print(ans)


if __name__ == '__main__':
    ans = []
    n_values = [20, 50, 100, 200, 500]  # n 的所有可能取值
    m_values = [5, 10, 20]  # m 的所有可能取值
    jobs_tuples = []
    seedData = generator.SeedData()
    for n in n_values:
        for m in m_values:
            seeds = seedData.get_seeds(n, m)
            if seeds is not None:
                jobs_tuples.extend((n, m, seed) for seed in seeds)
    # 参数列表
    params_list = [{'Pmax': 10, 'Smin': 0, 'Smax': 7, 'sigma_min': 0, 'sigma_max': 5,
                    'pls': 0.15, 'tuple': jobs_tuple, 'lambd': 10, 'x': 5, 'tmax': 30 * 60,
                    'cost_function': main.calculate_cost, 'ans': ans}
                   for jobs_tuple in jobs_tuples]

    # 创建一个进程池，使用所有可用的核心
    pool = multiprocessing.Pool()

    # 使用 map 函数并行运行进化算法
    results = pool.map(evolution_algorithm, params_list)

    # 关闭进程池
    pool.close()
    pool.join()

    # 处理结果
