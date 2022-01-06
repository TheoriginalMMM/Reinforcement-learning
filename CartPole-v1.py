import gym
import matplotlib 
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

matplotlib.use('TkAgg')
import memory
import matplotlib.pyplot as plt


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Initialisation of a buffer with a maximum size of 100
memoire = memory.Memory(100)

env = gym.make('CartPole-v1')
list_reward = {}
nb_action_episodes = {}
for i_episode in range(20):
    observation = env.reset()
    cumul_reward = 0
    nb_actions = 0
    for t in range(100):
        infos = []
        env.render()
        print(observation)
        print(type(observation))
        infos.append(observation)
        action = env.action_space.sample()
        infos.append(action)
        observation, reward, done, info = env.step(action)
        infos.append(observation)
        infos.append(reward)
        infos.append(done)
        cumul_reward += reward
        list_reward[t] = cumul_reward
        nb_actions+=1
        
        memoire.add_tupel(infos)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            nb_action_episodes[i_episode]=nb_actions
            break

plt.scatter(list(list_reward.keys()),list(list_reward.values()))
plt.show()


plt.scatter(list(nb_action_episodes.keys()),list(nb_action_episodes.values()))
plt.show()

# PARTIE 2.2 
# Étape 07 : Résseau de neurones pour l'environement Carte Pole v1 
# Couche d'entré  : Taille est égale au composante d'un état : 4
# Couche de sorte : Nombre d'action possible (2)
# Archetecture des couches caché : 4 : 32 : 2 
# Activation : 

