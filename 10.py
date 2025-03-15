import gym

# 使用 CartPole-v1 环境
env = gym.make('CartPole-v1', render_mode='human')
env.reset()
env.render()

# 后续可以添加更多的交互代码
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs = env.reset()

env.close()
