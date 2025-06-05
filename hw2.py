import random

cities = [
    (0, 3), (0, 0), (0, 2), (0, 1),
    (1, 0), (1, 3), (2, 0), (2, 3),
    (3, 0), (3, 3), (3, 1), (3, 2)
]

# 欧几里得距离
def euclidean_distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

# 路径总长
def total_distance(path):
    return sum(
        euclidean_distance(cities[path[i]], cities[path[(i+1) % len(path)]])
        for i in range(len(path))
    )


def generate_neighbor(path):
    new_path = path[:]
    i, j = sorted(random.sample(range(len(path)), 2))
    new_path[i:j+1] = reversed(new_path[i:j+1])
    return new_path

# 主体
def hill_climbing_tsp(initial_path, max_attempts=10000):
    current_path = initial_path
    current_cost = total_distance(current_path)

    attempts = 0
    while attempts < max_attempts:
        candidate = generate_neighbor(current_path)
        candidate_cost = total_distance(candidate)

        if candidate_cost < current_cost:
            current_path = candidate
            current_cost = candidate_cost
            attempts = 0  # 重置失败次数
        else:
            attempts += 1

    return current_path, current_cost

# 执行
initial = list(range(len(cities)))
random.shuffle(initial)

best_path, best_cost = hill_climbing_tsp(initial)
print("最优路径:", best_path)
print("路径总长:", best_cost)
