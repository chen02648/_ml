import gymnasium as gym
import numpy as np

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

for _ in range(10000):
    env.render()
    action = stable_policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if terminated or truncated:
        print(f"Episode {episodes + 1}: {steps} steps")
        total_steps += steps
        max_steps = max(max_steps, steps)
        episodes += 1
        steps = 0
        observation, info = env.reset()

env.close()
print(f"平均撐住：{total_steps // episodes} 步，最久撐住：{max_steps} 步")
