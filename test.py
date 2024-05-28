import sqlite3

import generator


def calculate_ARPD(C, C_min):
    R = len(C)
    return (1 / R) * sum(((Ci - C_min) / C_min) * 100 for Ci in C)

def calculate_minRPD(C, C_min):
    R = len(C)
    return min(((Ci - C_min) / C_min) * 100 for Ci in C)

def calculate_maxRPD(C, C_min):
    R = len(C)
    return max(((Ci - C_min) / C_min) * 100 for Ci in C)

# 示例用法
C = [25, 30, 35, 40]  # Ci 表示每个实例的完工时间
C_min = min(C)  # 最小完工时间
ARPD = calculate_ARPD(C, C_min)


conn = sqlite3.connect('D:\\google_buffer\\ans (1).db')  # 连接到 SQLite 数据库（如果不存在则会自动创建）
cursor = conn.cursor()  # 创建一个游标对象，用于执行 SQL 语句
# 执行查询
cursor.execute("select n,m,seed,makespan from results;")

results = cursor.fetchall()  # 如果你需要获取所有行的数据

# 关闭游标和连接
cursor.close()
conn.close()
best_solutions = {(20,5):1374,(20,10):1698,(20,20):2436,(50,5):2980,(50,10):3611,(50,20):4479,(100,5):6065,(100,10):6906,
(100,20):7709,(200,10):13166,(200,20):14516,(500,20):35380}
ans = {}
# 如果需要获取所有行的数据
for row in results:
    n, m, seed,makespan = row
    if (n,m,seed) in ans:
        ans[(n,m,seed)].append(makespan)
    else:
        ans[(n,m,seed)] = [makespan]
mapping = {(20,5):1,(20,10):11,(20,20):21,(50,5):31,(50,10):41,(50,20):51,(100,5):61,(100,10):71,
(100,20):81,(200,10):91,(200,20):101,(500,20):111}
# 按元组的第一个元素从小到大进行排序
n=500
m=20
l = {}
sorted_data = sorted(ans.items(), key=lambda x: x[0])
for key, value in sorted_data:
    if key[0]==n and key[1] == m:
        l[key[2]]=min(value)
    '''print(
          round(calculate_minRPD(value,best_solutions[key]),3),'\t',
          round(calculate_maxRPD(value,best_solutions[key]),3),'\t',
          round(calculate_ARPD(value,best_solutions[key]),3))'''
    # 在这里可以使用 column1_value 和 column2_value 做任何你想做的事情
# 给定的顺序列表
a = generator.SeedData()
sorted_keys = a.get_seeds(n,m)

print(sorted_keys)
for i in sorted_keys:
    print(l[i])
