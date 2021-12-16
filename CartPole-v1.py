import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
list_reward = {}
for i_episode in range(20):
    observation = env.reset()
    cumul_reward = 0
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        cumul_reward += reward
        list_reward[t] = cumul_reward
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()

plt.scatter(list(list_reward.keys()),list(list_reward.values()))
plt.show()
