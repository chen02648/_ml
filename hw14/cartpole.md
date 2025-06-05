
採用以下加權線性控制公式：

```
control = k1 * x + k2 * dx + k3 * theta + k4 * dtheta + 補償項
```

* 根據 pole angle (`theta`) 大小調整增益：

  * 角度越小 → 控制微調
  * 角度越大 → 控制放大（避免來不及修正）

* 當車子位置偏移過多（如超出 ±1.5）時，加入額外 "position correction" 補償推力

* 最終決策以 `control < 0` 向左推，`control >= 0` 向右推

---

### 程式碼

```python
import gymnasium as gym
import numpy as np
import csv

def stable_policy(observation):
    x, dx, theta, dtheta = observation

    if abs(theta) < 0.05:
        k_x, k_dx, k_theta, k_dtheta = 1.0, 0.3, 2.5, 1.0
    elif abs(theta) < 0.15:
        k_x, k_dx, k_theta, k_dtheta = 1.0, 0.5, 3.5, 1.5
    else:
        k_x, k_dx, k_theta, k_dtheta = 1.2, 0.7, 4.5, 2.0

    pos_penalty = 1.5 * np.sign(x) if abs(x) > 1.5 else 0

    control = (k_x * x + k_dx * dx + k_theta * theta + k_dtheta * dtheta) + pos_penalty
    return 0 if control < 0 else 1

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

steps = 0
episodes = 0
total_steps = 0
max_steps = 0
record = []

for _ in range(10000):
    env.render()
    action = stable_policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if terminated or truncated:
        print(f"Episode {episodes + 1}: {steps} steps")
        record.append([episodes + 1, steps])
        total_steps += steps
        max_steps = max(max_steps, steps)
        episodes += 1
        steps = 0
        observation, info = env.reset()

env.close()

with open("cartpole_strategy_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Steps Survived"])
    writer.writerows(record)

print(f"平均撐住：{total_steps // episodes} 步，最久撐住：{max_steps} 步")
```

---

###  輸出紀錄（`cartpole_strategy_log.csv`）

| Episode | Steps Survived |
| ------- | -------------- |
| 1       | 226            |
| 2       | 309            |
| 3       | 284            |
| ...     | ...            |

> 可用於畫圖、報告附錄、比較不同策略表現

