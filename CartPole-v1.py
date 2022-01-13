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


#PARAMETERS 
# entre 0 et 1 
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10
# Quelle méthode de MAJ des paramtres du résseaux de neurones target
# Chaque étape _ True 
TARGET_UPDATE_EACH_STEP = False
# ALpha utilisé pour la mAJ des paramtres
ALPHA = 0.01
# A quelle fréqeuence modifié les paramtres
TARGET_UPDATE_FREQ = 500



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

        states = torch.as_tensor(np.asarray([e.state for e in experiences if e is not None]), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray([e.action for e in experiences if e is not None]), dtype=torch.int64).unsqueeze(-1)
        rewards = torch.as_tensor(np.asarray([e.reward for e in experiences if e is not None]),dtype=torch.float32).unsqueeze(-1)
        next_states = torch.as_tensor(np.asarray([e.next_state for e in experiences if e is not None]),dtype=torch.float32)
        dones = torch.as_tensor(np.asarray([e.done for e in experiences if e is not None]),dtype=torch.float32).unsqueeze(-1)

        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

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
    def __init__(self, action_dim, state_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, action_dim)

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
def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
#    if is_ipython:
#        display.clear_output(wait=True)
#        display.display(plt.gcf())


# A partir de maintenant on va essayer de rassembler tous les bouts de code dans une classe d'agent #######
#############################################################################
## Partie 2.3 Exploration
class AgentExploration:
    
    
    def __init__(self,gamma,epsilon = 0.99,beta = 0.99,batch_size=20 ):
        self.gamma = gamma
        self.epsilon=epsilon
        self.beta = beta
        self.env= gym.make('CartPole-v1')


        self.online_network = QNetwork(self.env.action_space.n,len(self.env.reset()))
        self.target_network = QNetwork(self.env.action_space.n,len(self.env.reset()))
        self.optim = torch.optim.Adam(self.online_network.parameters(), lr=0.001)
        # Initialisationa avec les même paramtres
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.batch_size = BATCH_SIZE
        self.memory=ReplayBuffer(BUFFER_SIZE,self.batch_size,1)
        self.compteur_train= 0
        

    def choose_action(self,state):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = self.online_network(state)

    # select a random action wih probability eps
        if random.random() <= self.epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = np.argmax(values.cpu().numpy())
        return action

    def update_epsilon(self):
        self.epsilon=self.epsilon*self.beta

    def update_target_network_weights_each_step(self,):
        print("updating target netowrk paramaters ")
        for param_tensor in self.online_network.state_dict():
            self.target_network.state_dict()[param_tensor] = (1-ALPHA)*self.target_network.state_dict()[param_tensor]+self.online_network.state_dict()[param_tensor]*ALPHA

    
    def run(self):
        
        #env = gym.wrappers.Monitor(env, "recording",force=True)
        observationN = self.env.reset()
        episode_durations = []
        list_reward = {}
        cumul_reward = 0
        for i_episode in range(100):
            observationN = self.env.reset()
            cumul_reward = 0
            nb_actions = 0
            for t in range(100):
                self.env.render()      
                # Choix de l'action
                action = self.choose_action(observationN)
                print("ACTION : ",action)
                observationF, reward, done, info = self.env.step(action)
                
                cumul_reward += reward
                nb_actions+=1
                list_reward[t] = cumul_reward
                self.memory.add(observationN,action,reward,observationF,done)
                
                #self.memory.push(observationN,action,observationF,reward,done)
                observationN=observationF
                
                if self.memory.__len__ () > self.batch_size:
                    self.train(self.batch_size,self.optim,self.memory)
                
                # exploration alpha greedy avec decay
                self.update_epsilon()
               

                if done:
                    plot_durations(episode_durations)
                    print("Episode finished after {} timesteps".format(t + 1))
                    episode_durations.append(t + 1)
                    break
            
        plt.scatter(list(list_reward.keys()),list(list_reward.values()))
        plt.show()

    
    def train(self,batch_size, optim, memory ):
        self.compteur_train +=1 
        # Juste avec un seul model 
        resultats = self.memory.sample()
        # On calcule les Q valeurs estimé par le premier réseau de neurones
        
        #target q values : 
        target_q_values = self.target_network(resultats[3])

        #MAX TARGET Q VALUE 
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = resultats[2] + self.gamma * max_target_q_values * (1 - resultats[4])


        q_values = self.online_network(resultats[0])
        
        action_q_values = torch.gather(input=q_values, dim=1, index=resultats[1])

        loss = nn.MSELoss()
        loss=loss(action_q_values ,targets)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Updating target Network params
        if TARGET_UPDATE_EACH_STEP:
            self.update_target_network_weights_each_step()
        else:
            if self.compteur_train %TARGET_UPDATE_FREQ == 0:
                self.target_network.load_state_dict(self.online_network.state_dict())
                

ae = AgentExploration(0.5)
ae.run()


########################## 2.4 Apprentissage ############################



