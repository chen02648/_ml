import random

def f(x, y, z):
    return (x - 1)**2 + (y - 2)**2 + (z - 3)**2

def hill_climb(iterations=1000, step_size=0.1):
    x, y, z = random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)
    current_value = f(x, y, z)

    for _ in range(iterations):
        neighbors = [
            (x + step_size, y, z), (x - step_size, y, z),
            (x, y + step_size, z), (x, y - step_size, z),
            (x, y, z + step_size), (x, y, z - step_size)
        ]
        
        next_point = min(neighbors, key=lambda p: f(*p))
        next_value = f(*next_point)
        
        if next_value < current_value:
            x, y, z = next_point
            current_value = next_value
        else:
            break  

    return (x, y, z), current_value

best_point, best_value = hill_climb()
print("找到的最低点:", best_point)
print("函数值:", best_value)
