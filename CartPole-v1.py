import gym
import matplotlib 
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

# Initiaiton avec GYM et CartPole-v1
# #########################################
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

# Deuxiemme maniere pour tester l'apprentissage.
########################################################
# Compter le nombre d'action qui fait pour chaque épisode
#########################################################
#plt.scatter(list(nb_action_episodes.keys()),list(nb_action_episodes.values()))
#plt.show()

#2.1 Experience replay:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self,buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

######################################################################################
# test_memory = ReplayBuffer(20,10,0)
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
        
#         #test_memory.push(observationN,action,observationF,reward,done)
#         test_memory.add(observationN,action,reward,observationF,done)
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             nb_action_episodes[i_episode]=nb_actions
#             break

# print(test_memory.__len__())
# resultats = test_memory.sample()
# print(len(resultats))
# print(type(resultats[:][0]))
# print("end test")
######################################################################################
######################################################################################
# PARTIE 2.2 
# Étape 07 : Résseau de neurones pour l'environement Carte Pole v1 
# Couche d'entré  : Taille est égale au composante d'un état : 4
# Couche de sorte : Nombre d'action possible (2)
# Archetecture des couches caché : 4 : 32 : 2 
# Activation :  Leaky relu 
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
######################################################################
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
#######################################################################

# A partir de maintenant on va essayer de rassembler tous les bouts de code dans une classe d'agent #######
#############################################################################
## Partie 2.3 Exploration
class AgentExploration:
    
    
    def __init__(self,gamma,epsilon = 0.99,beta = 0.99,batch_size=5 ):
        self.gamma = gamma
        self.epsilon=epsilon
        self.memory = None
        self.beta = beta
        self.env= gym.make('CartPole-v1')
        self.model = QNetwork(self.env.action_space.n,len(self.env.reset()),8)
        self.batch_size = 5
        self.memory=ReplayBuffer(100,self.batch_size,0)
        

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

        list_reward = {}
        cumul_reward = 0
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
                list_reward[t] = cumul_reward
                self.memory.add(observationN,action,reward,observationF,done)
                
                #self.memory.push(observationN,action,observationF,reward,done)
                observationN=observationF
                if self.memory.__len__ () > self.batch_size:
                    self.train(self.batch_size,optim,self.memory)
                self.update_epsilon()
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

        plt.scatter(list(list_reward.keys()),list(list_reward.values()))
        plt.show()

    
    def train(self,batch_size, optim, memory ):
        # Juste avec un seul model 
        resultats = self.memory.sample()
        # On calcule les Q valeurs estimé par le premier réseau de neurones
        q_values = self.model(resultats[0])
        next_q_values = self.model(resultats[3])
        q_value = q_values.gather(1, resultats[1])
        next_q_value = torch.max(next_q_values, 1)[0].detach()
        expected_q_value = resultats[2] + self.gamma * next_q_value * (1 - resultats[4])
        loss = nn.MSELoss()
        loss=loss(q_value ,expected_q_value)
        optim.zero_grad()
        loss.backward()
        optim.step()

ae = AgentExploration(0.5)
ae.run()


########################## 2.4 Apprentissage ############################



