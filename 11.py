import gym
import time

# 去掉环境 ID 中的空格
env = gym.make('CartPole-v1', render_mode='human')
# 重置环境
observation = env.reset()
# 如果返回值是元组，取第一个元素作为观测值
if isinstance(observation, tuple):
    observation = observation[0]
count = 0

for t in range(100):
    # 随机采样一个动作
    action = env.action_space.sample()
    # 执行动作，接收 5 个返回值
    observation, reward, terminated, truncated, info = env.step(action)
    # 判断是否结束
    done = terminated or truncated
    if done:
        break
    # 渲染环境
    env.render()
    count += 1
    time.sleep(0.2)

print(count)
# 关闭环境
env.close()