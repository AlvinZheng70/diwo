import math

import numpy as np


class SeedData:
    def __init__(self):
        self.data = {
            (20, 5): [873654221, 379008056, 1866992158, 216771124, 495070989,
                      402959317, 1369363414, 2021925980, 573109518, 88325120],
            (20, 10): [587595453, 1401007982, 873136276, 268827376, 1634173168,
                       691823909, 73807235, 1273398721, 2065119309, 1672900551],
            (20, 20): [479340445, 268827376, 1958948863, 918272953, 555010963,
                       2010851491, 1519833303, 1748670931, 1923497586, 1829909967],
            (50, 5): [1328042058, 200382020, 496319842, 1203030903, 1730708564,
                      450926852, 1303135678, 1273398721, 587288402, 248421594],
            (50, 10): [1958948863, 575633267, 655816003, 1977864101, 93805469,
                       1803345551, 49612559, 1899802599, 2013025619, 578962478],
            (50, 20): [1539989115, 691823909, 655816003, 1315102446, 1949668355,
                       1923497586, 1805594913, 1861070898, 715643788, 464843328],
            (100, 5): [896678084, 1179439976, 1122278347, 416756875, 267829958,
                       1835213917, 1328833962, 1418570761, 161033112, 304212574],
            (100, 10): [1539989115, 655816003, 960914243, 1915696806, 2013025619,
                        1168140026, 1923497586, 167698528, 1528387973, 993794175],
            (100, 20): [1368624604, 450181436, 1927888393, 1759567256, 606425239,
                        19268348, 1298201670, 2041736264, 379756761, 28837162],
            (200, 10): [471503978, 1215892992, 135346136, 1602504050, 160037322,
                        551454346, 519485142, 383947510, 1968171878, 540872513],
            (200, 20): [2013025619, 475051709, 914834335, 810642687, 1019331795,
                        2056065863, 1342855162, 1325809384, 1988803007, 765656702],
            (500, 20): [1368624604, 450181436, 1927888393, 1759567256, 606425239,
                        19268348, 1298201670, 2041736264, 379756761, 28837162],
        }

    def get_seeds(self, jobs, machines):
        return self.data.get((jobs, machines), None)


class LinearCongruentialGenerator:
    def __init__(self, seed):
        self.seed = seed
        self.a = 16807
        self.b = 127773
        self.c = 2836
        self.m = 2 ** 31 - 1

    def generate(self):
        k = self.seed // self.b  # 整数除法运算
        self.seed = self.a * (self.seed % self.b) - k * self.c
        if self.seed < 0:
            self.seed += self.m
        return self.seed / self.m

    def generate_uniform(self, a, b):
        return math.floor(a + self.generate() * (b - a + 1))


def calculate_lb(matrix):
    m = len(matrix[0])
    lb_values = []
    for i in range(m):
        b_i = 0 if i == 0 else np.min(np.sum(matrix[:, :i], axis=1))
        a_i = 0 if i == m - 1 else np.min(np.sum(matrix[:, i + 1:], axis=1))
        T_i = np.sum(matrix[:, i])
        lb_values.append(b_i + T_i + a_i)
    return max(np.max(lb_values), np.max(np.sum(matrix[:, :], axis=1)))


def generate_processing_times(seed, m, n):
    """
    生成处理时间矩阵
    :param seed: 初始化种子
    :param m: 机器数
    :param n: 工件数
    :return: 处理时间矩阵，LB（下界）
    """
    generator = LinearCongruentialGenerator(seed=seed)
    processing_times = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            processing_times[i][j] = generator.generate_uniform(1, 99)
    processing_times = np.transpose(processing_times)
    return processing_times, calculate_lb(processing_times)
