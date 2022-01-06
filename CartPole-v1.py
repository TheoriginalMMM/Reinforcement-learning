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
import random
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

# env = gym.make('CartPole-v1')
# list_reward = {}
# nb_action_episodes = {}
# for i_episode in range(20):
#     observation = env.reset()
#     cumul_reward = 0
#     nb_actions = 0
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         cumul_reward += reward
#         list_reward[t] = cumul_reward
#         nb_actions+=1

#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             nb_action_episodes[i_episode]=nb_actions
#             break

#plt.scatter(list(list_reward.keys()),list(list_reward.values()))
#plt.show()


#plt.scatter(list(nb_action_episodes.keys()),list(nb_action_episodes.values()))
#plt.show()

#2.1 Experience replay:

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward','done'))

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

# Test : 
# test_memory = ReplayMemory(20)
# env = gym.make('CartPole-v1')
# env = gym.wrappers.Monitor(env, "recording",force=True)
# list_reward = {}
# nb_action_episodes = {}
# for i_episode in range(20):
#     observationN = env.reset()
#     cumul_reward = 0
#     nb_actions = 0
#     for t in range(100):

#         env.render()        
#         action = env.action_space.sample()

#         observationF, reward, done, info = env.step(action)
        

#         cumul_reward += reward
#         list_reward[t] = cumul_reward
#         nb_actions+=1
        
#         test_memory.push(observationN,action,observationF,reward,done)

#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             nb_action_episodes[i_episode]=nb_actions
#             break

# print(test_memory.__len__())
# resultats = test_memory.sample(10)
# print(len(resultats))

# PARTIE 2.2 
# Étape 07 : Résseau de neurones pour l'environement Carte Pole v1 
# Couche d'entré  : Taille est égale au composante d'un état : 4
# Couche de sorte : Nombre d'action possible (2)
# Archetecture des couches caché : 4 : 32 : 2 
# Activation : 

# Réponse 07: Archtecture proposé 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1

# Réponse 08 : 
#
# model : le QNetwork
# env : l'environement 
# state l'etat acctuel 
#
def select_action(model, state):
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)
    print(values.cpu().numpy())
    action = np.argmax(values.cpu().numpy())

    return action

## TEST DES DERNIERS FONCTIONS :
# test_memory = ReplayMemory(20)
# env = gym.make('CartPole-v1')
# #env = gym.wrappers.Monitor(env, "recording",force=True)
# observationN = env.reset()
# print("ACTION SPACE ",env.action_space.n)
# model = QNetwork(env.action_space.n,len(observationN),8)
# for i_episode in range(20):
#     observationN = env.reset()
#     cumul_reward = 0
#     nb_actions = 0
#     for t in range(100):
#         env.render()      
#         # Choix de l'action
#         action = select_action(model,observationN)
#         print("ACTION : ",action)
#         observationF, reward, done, info = env.step(action)
#         cumul_reward += reward
#         nb_actions+=1
        
#         test_memory.push(observationN,action,observationF,reward,done)
#         observationN=observationF
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break

## Partie 2.3 Exploration
class AgentExploration:
    def __init__(self,gamma,epsilon = 0.99,beta = 0.99, ):
        self.gamma = gamma
        self.epsilon=epsilon
        self.memory = None
        self.beta = beta
        self.env= gym.make('CartPole-v1')
        self.model = QNetwork(self.env.action_space.n,len(self.env.reset()),8)
        self.memory=ReplayMemory(1000)
        self.batch_size = 5

    def choose_action(self,model,env,state):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = model(state)

    # select a random action wih probability eps
        if random.random() <= self.epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(values.cpu().numpy())
        return action

    def update_epsilon(self):
        self.epsilon=self.epsilon*self.beta
    
    def run(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        env = gym.make('CartPole-v1')
        #env = gym.wrappers.Monitor(env, "recording",force=True)
        observationN = env.reset()
        print("ACTION SPACE ",env.action_space.n)
        model = QNetwork(env.action_space.n,len(observationN),8)
        for i_episode in range(20):
            observationN = env.reset()
            cumul_reward = 0
            nb_actions = 0
            for t in range(100):
                env.render()      
                # Choix de l'action
                action = self.choose_action(model,env,observationN)
                print("ACTION : ",action)
                observationF, reward, done, info = env.step(action)
                
                

                cumul_reward += reward
                nb_actions+=1
                
                self.memory.push(observationN,action,observationF,reward,done)
                observationN=observationF
                if self.memory.__len__ ()>= self.batch_size:
                    self.train(self.batch_size,optim,self.memory)
                self.update_epsilon()
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

    
    def train(self,batch_size, optim, memory ):

        states, actions, next_states, rewards, is_done = self.memory.sample(self.batch_size)
        print(type(states()))
        states = torch.Tensor(states).to(device)
        
        q_values = self.model(states)
        next_states = torch.Tensor(next_states).to(device)
        next_q_values = self.model(next_states)
        

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_value = torch.max(next_q_values, 1)
        
        expected_q_value = rewards + self.gamma * next_q_value * (1 - is_done)

        loss = nn.MSELoss()
        loss=loss(q_value ,expected_q_value)
        optim.zero_grad()
        loss.backward()
        optim.step()

ae = AgentExploration(0.5)
ae.run()


########################## 2.4 Apprentissage ############################



