from ast import Try
from distutils.fancy_getopt import FancyGetopt
from pickle import TRUE
from tokenize import Triple
import gym
from gym.wrappers import Monitor
from gym.wrappers.monitoring import video_recorder


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

##################################################################################
#PARAMETERS ET CONFIGURATIONS

TRAIN = False
TEST = True
SAVE_MODEL_PARAMS = False
LOAD_MODEL_PARAMS = True
SAVE_HYPER_PARAMS = False

MODEL_PARAMS_PATH = "CartPole.data"
HYPER_PARAMS_PATH = "Hyper_Params.txt"
#400
NB_EPISODES_TRAIN = 500
NB_EPISODES_TEST = 1
START_TRAIN = 1000

RECORD_PERFS = False
RENDRING_ENV = True

LEARNING_RATE = 0.00001
# 0.9
GAMMA = 0.99

EPSILON_START = 0.99
#0.01
EPSILON_END = 0.025
EPSILON_DECAY = 0.999

BATCH_SIZE = 512
BUFFER_SIZE = 20000

TARGET_PARAMS_UPDATE_EACH_STEP = False
ALPHA = 0.01

# 100
TARGET_UPDATE_FREQ = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


HPARAMS = {
    "np episodes train":NB_EPISODES_TRAIN,
    "start train": START_TRAIN,
    "lr":LEARNING_RATE,
    "gamma":GAMMA,
    "epsi start ":EPSILON_START,
    "epsi end ": EPSILON_END,
    "epsi decay": EPSILON_DECAY,
    "batch_size": BATCH_SIZE,
    "buffer_size":BUFFER_SIZE,
    "target update each step":TARGET_PARAMS_UPDATE_EACH_STEP,
    "alph update ":ALPHA,
    "target update freq":TARGET_UPDATE_FREQ
}

if SAVE_HYPER_PARAMS : 
    with open(HYPER_PARAMS_PATH, "w") as fichier:
        for i in HPARAMS.keys():
            fichier.write(f"{i} : {HPARAMS[i]} \n")

##################################################################################
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

def plot_evolution(episode_durations,phase):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title(phase+'...')
    plt.xlabel('Episode')
    plt.ylabel('Récompenses')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())


    
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig(phase+".png")


#2.1 Experience replay:
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

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

######################################################################################
######################################################################################
# PARTIE 2.2 
# Étape 07 : Résseau de neurones pour l'environement Carte Pole v1 
# Couche d'entré  : Taille est égale au composante d'un état : 4
# Couche de sorte : Nombre d'action possible (2)
# Archetecture des couches caché : 4 : 32 : 2 
# Activation :  Leaky relu 
# Réponse 07: Archtecture proposé 

"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, action_dim)

        # self.fc_1 = nn.Linear(state_dim, 32)
        # self.fc_2 = nn.Linear(32, action_dim)

    def forward(self, inp):
        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        # x1 = F.leaky_relu(self.fc_1(inp))
        # x1 = self.fc_2(x1)
        return x1
###########################################################################
# Réponse 08 : 
#
# model : le QNetwork
# env : l'environement 
# state l'etat acctuel 
####################

def select_action(model, state):
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)
    print(values.cpu().numpy())
    action = np.argmax(values.cpu().numpy())

    return action

# A partir de maintenant on va essayer de rassembler tous les bouts de code dans une classe d'agent #######
#############################################################################
## Partie 2.3 Exploration

class DQNAgent:
    
    def __init__(self,env,exploration= True ,gamma=GAMMA,epsilon_start= EPSILON_START,epsilon_decay = EPSILON_DECAY,replay_memory_size=BUFFER_SIZE,batch_size=BATCH_SIZE):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.env= env
        self.exploration = exploration
        self.batch_size = batch_size
        self.memory=ReplayBuffer(replay_memory_size,self.batch_size,0)
        self.compteur_train= 0

        self.online_network = QNetwork(self.env.action_space.n,len(self.env.reset()))
        self.target_network = QNetwork(self.env.action_space.n,len(self.env.reset()))
        self.optim = torch.optim.Adam(self.online_network.parameters(), lr=LEARNING_RATE)
        
        if LOAD_MODEL_PARAMS :
            self.online_network.load_state_dict(torch.load(MODEL_PARAMS_PATH))
        
        # Initialisationa avec les même paramtres
        self.target_network.load_state_dict(self.online_network.state_dict())
        # seting seed to 0
        self.env.seed(0)


    def choose_action(self,state):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = self.online_network(state)

        # select a random action wih probability eps
        if self.exploration :
                
                if random.random() <= self.epsilon:
                    #print("RANDOM ACTION")
                    action = np.random.randint(0, self.env.action_space.n)
                else:
                    action = np.argmax(values.cpu().numpy())
        else:
            action = np.argmax(values.cpu().numpy())

        return action

    def update_epsilon(self):
        self.epsilon=self.epsilon*self.epsilon_decay
        self.epsilon = max(self.epsilon, EPSILON_END)
        #print(f"Updating Epsilon after {self.compteur_train} New epsilon : {self.epsilon}")

    def update_target_network_weights_each_step(self,):
        print("updating target netowrk paramaters ")
        Online_PARAMS = self.online_network.state_dict()
        Target_PARAMS = self.target_network.state_dict()

        for i in Online_PARAMS:
            Target_PARAMS[i]=(1-ALPHA)*Target_PARAMS[i]+ALPHA*Online_PARAMS[i]

        self.target_network.load_state_dict(Target_PARAMS)
    def run(self):
        if TRAIN:
            observationN = self.env.reset()
            episode_durations = []
            list_reward = {}
            for i_episode in range(NB_EPISODES_TRAIN):
                
                observationN = self.env.reset()
                cumul_reward = 0
                nb_actions = 0
                done = False
                #for t in range(MAX_ACTIONS_PER_EPISODES):
                while not done:
                    # if RENDRING_ENV : self.env.render()      
                    # Choix de l'action
                    action = self.choose_action(observationN)
                    observationF, reward, done, info = self.env.step(action)
                    
                    cumul_reward += reward
                    nb_actions+=1
                    self.memory.add(observationN,action,reward,observationF,done)

                    observationN=observationF
                    
                    if self.memory.__len__ () > BATCH_SIZE:
                        self.learn(self.optim,self.memory)
                    
                    
                    # exploration alpha greedy avec decay
                    if self.exploration :
                        self.update_epsilon()

                    if done:
                        list_reward[i_episode] = cumul_reward
                        plot_evolution(episode_durations,'Train')
                        #print("Episode finished after {} timesteps".format(nb_actions))
                        episode_durations.append(nb_actions)
                        #print("CUMUL REWARDS FOR EPISODE ",i_episode,"is :",cumul_reward)
                        break
            
            if SAVE_MODEL_PARAMS:
                torch.save(self.online_network.state_dict(), MODEL_PARAMS_PATH)


        if TEST:
            observationN = self.env.reset()
            episode_durations = []
            list_reward = {}
            # Pour ne plus explorer
            if RECORD_PERFS:
                video_path='demo.mp4'
                video_recorder_object =video_recorder.VideoRecorder(self.env, path=video_path)

            self.exploration = False
            for i_episode in range(NB_EPISODES_TEST):
                observationN = self.env.reset()
                cumul_reward = 0
                nb_actions = 0
                done = False
                #for t in range(MAX_ACTIONS_PER_EPISODES):
                while not done:
                    if RENDRING_ENV : 
                        self.env.render()
                    if RECORD_PERFS:
                        video_recorder_object.capture_frame()      
                    # Choix de l'action
                    action = self.choose_action(observationN)
                    observationF, reward, done, info = self.env.step(action)

                    cumul_reward += reward
                    nb_actions+=1                    
                    #self.memory.push(observationN,action,observationF,reward,done)
                    observationN=observationF
                    
                    # exploration alpha greedy avec decay
                    if self.exploration :
                        self.update_epsilon()
                    if done:
                        list_reward[i_episode] = cumul_reward
                        episode_durations.append(nb_actions)
                        plot_evolution(episode_durations,"Test")

                        #print("Episode finished after {} timesteps".format(nb_actions))
                        #print("CUMUL REWARDS FOR EPISODE ",i_episode,"is :",cumul_reward)
                        break
            if RECORD_PERFS:
                video_recorder_object.close()
                video_recorder_object.enabled = False
            
            moy = np.mean(list(list_reward.values()))
            ecart_type= np.std(list(list_reward.values()))
            print(f" Moyenne +/- Ecart type  des récompenses sur les {NB_EPISODES_TEST} episodes")
            print(f" Moyennes : {moy}")
            print(f" Ecart type  : {ecart_type}")
            self.env.close()

    
    def learn(self, optim, memory ):
        self.compteur_train +=1 
        # Juste avec un seul model 
        resultats = self.memory.sample()

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
        if TARGET_PARAMS_UPDATE_EACH_STEP:
            self.update_target_network_weights_each_step()
        elif self.compteur_train % TARGET_UPDATE_FREQ == 0 :
                print(f"COMPTEUR {self.compteur_train} UPDATE PARAMS EPSILON {self.epsilon} ")
                self.target_network.load_state_dict(self.online_network.state_dict())



                
env = gym.make('CartPole-v1')
ae = DQNAgent(env)
ae.run()



