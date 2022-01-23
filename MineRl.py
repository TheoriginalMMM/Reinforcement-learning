
from typing import Tuple
import matplotlib 
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import gym
from gym.spaces import Box
from frame_stacking import FrameStack
from gym.wrappers import Monitor





import environment

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import transforms as T
from gym.wrappers.monitoring import video_recorder


matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#PARAMETERS ET CONFIGURATIONS

TRAIN = True
TEST = True
SAVE_MODEL_PARAMS = True
LOAD_MODEL_PARAMS = False

MODEL_PARAMS_PATH = "MineRl-FrameStacking.data"
HYPER_PARAMS_PATH = "MineRL-Hyper_Params-FrameStacking.txt"
#400
NB_EPISODES_TRAIN = 200
MAX_ACTIONS_PER_EPISODES = 300
NB_EPISODES_TEST = 1
START_TRAIN = 1000

RECORD_PERFS = True
RENDRING_ENV = True
FRAME_STACKING = True

LEARNING_RATE = 0.0001
# 0.9
GAMMA = 0.99

EPSILON_START = 0.99
#0.01
EPSILON_END = 0.025
EPSILON_DECAY = 0.9999

BATCH_SIZE = 250
BUFFER_SIZE = 20000

TARGET_PARAMS_UPDATE_EACH_STEP = False
ALPHA = 0.01

# 100
TARGET_UPDATE_FREQ = 40

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

# Sauvgarder la liste des paramtres 
# with open(HYPER_PARAMS_PATH, "w") as fichier:
#     for i in HPARAMS.keys():
# 	    fichier.write(f"{i} : {HPARAMS[i]} \n")

# 
SIMPLE_KEYBOARD_ACTION ={'Mineline-v0': ['left', 'right', 'attack']}

def plot_evolution(episode_durations,phase):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title(phase+'...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())


    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if not RECORD_PERFS:
        plt.savefig(phase+".png")

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

        states = [e.state for e in experiences if e is not None]
        actions = torch.as_tensor(np.asarray([e.action for e in experiences if e is not None]).astype(int), dtype=torch.int64).unsqueeze(-1)
        rewards = torch.as_tensor(np.asarray([e.reward for e in experiences if e is not None]),dtype=torch.float32).unsqueeze(-1)
        next_states = [e.next_state for e in experiences if e is not None]
        dones = torch.as_tensor(np.asarray([e.done for e in experiences if e is not None]),dtype=torch.float32).unsqueeze(-1)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

##################################################################################
# Initiaiton avec Mine RL 
# ##############################################################

# env = gym.make('Mineline-v0')
# obs = env.reset()
# print(obs['pov'])

# print(env.action_space)
# action = env.action_space.noop()
# action['right'] = 0
# action['left'] = 1
# obs, reward, done, _ = env.step(action)
# print("NEW FRAME")
# print("SANS PRE TRAITEMENT ")
# print(type(obs['pov']))
# print(len(obs['pov']))
# print(obs['pov'].shape)
##########################################################################

# Étape prétraitement des observations
# Changement de résolution 
# Noir et blanc
# Normalisation 
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space['pov'].shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation['pov'])
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

# Function pour associé les wrapper a notre environement
env_name ='Mineline-v0'
def make_env(name):
    env = gym.make(name)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    
    if FRAME_STACKING:
        env = FrameStack(env, 4)
    if RECORD_PERFS:
        #env = Monitor(env, directory="demoMineRl-monitor.mp4",force=True)
        print("ok")

    return env


# Fuction pour l'encodage des actions : 

def encode_actions(env_name):
    resultats = {}
    cmp = 0 
    for key in SIMPLE_KEYBOARD_ACTION[env_name]:
        #print(key)
        resultats[cmp] = key
        resultats[key] = cmp
        cmp+=1
    
    #print(f" encodage des actions finals : {resultats} ")
    return resultats




# Testons nos wrapper de pré traitement
# memory = ReplayBuffer(20,2,0)

# env = gym.make('Mineline-v0')
# env = GrayScaleObservation(env)
# env = ResizeObservation(env, shape=84)
# # env = ActionWrapper(env)

# #env = FrameStack(env,4)
# print("SA PASSE !")


# obs_i = env.reset()
# print(obs_i)

# for i in range(4):
#     # Initialitation of the action
#     print(env.action_space)
#     action = env.action_space.noop()
#     # Hwo to choos a action 
#     #action['right'] = 0
#     #action['left'] = 1
#     # Hwo to take a random action 
#     action = env.action_space.sample()
#     obs, reward, done, _ = env.step(action)
#     action = list(action.values())
#     memory.add(obs_i,action,reward,obs,done)
#     obs_i = obs
#     print(env.observation_space)
#     print("NEW FRAME")
#     print("Avec PRE TRAITEMENT ")
#     print(type(obs))
#     print(obs.shape)
#     if done:
#         break
   
# batch = memory.sample()
# print(batch[0])



        

######################################################################################
######################################################################################
# Résseau de neurones convolutifs 
# Couche d'entré  : Taille est égale au composante d'un état : 4
# Couche de sorte : Nombre d'action possible (2)
# Archetecture des couches caché : 4 : 32 : 2 
# Activation :  Leaky relu 
# Réponse 07: Archtecture proposé 

"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
"""
If the observations are images we use CNNs.
"""

class QNetworkCNN(nn.Module):
    def __init__(self, action_dim):
        super(QNetworkCNN, self).__init__()
        if FRAME_STACKING:
            # On met 4 chanel en entré
            self.conv_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        else:
            self.conv_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
            
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, action_dim)

    def forward(self, inp):
        x1 = F.relu(self.conv_1(inp))
        x1 = F.relu(self.conv_2(x1))
        x1 = F.relu(self.conv_3(x1))
        x1 = torch.flatten(x1, 1)
        x1 = F.leaky_relu(self.fc_1(x1))
        x1 = self.fc_2(x1)

        return x1



#########################################################################
class DQNAgent:
    
    def __init__(self,env_name,exploration= True ,gamma=GAMMA,epsilon_start= EPSILON_START,epsilon_decay = EPSILON_DECAY,replay_memory_size=BUFFER_SIZE,batch_size=BATCH_SIZE):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.env= make_env(env_name)
        self.action_codes = encode_actions(env_name)
        self.exploration = exploration
        self.batch_size = batch_size
        self.memory=ReplayBuffer(replay_memory_size,self.batch_size,0)
        self.compteur_train= 0

        self.nb_possible_action = len(SIMPLE_KEYBOARD_ACTION[env_name])

        self.online_network = QNetworkCNN(len(SIMPLE_KEYBOARD_ACTION[env_name]))
        self.target_network = QNetworkCNN(len(SIMPLE_KEYBOARD_ACTION[env_name]))
        self.optim = torch.optim.Adam(self.online_network.parameters(), lr=LEARNING_RATE)
        
        if LOAD_MODEL_PARAMS :
            self.online_network.load_state_dict(torch.load(MODEL_PARAMS_PATH))
        
        # Initialisationa avec les même paramtres
        self.target_network.load_state_dict(self.online_network.state_dict())
        # seting seed to 0
        self.env.seed(0)

    def get_code_from_env_action(self, action):
        for key, value in action.items():

            if value==1:
                #print(f"key {key } fasse to {self.action_codes[key]}")
                return self.action_codes[key]

    def choose_action(self,state):
        # etape obligatoir d'initialisation 
        if not FRAME_STACKING:
            # etape obligatoir d'initialisation 
            state = state.unsqueeze(0)
            state = state.unsqueeze(0)
        else:     
            state = torch.FloatTensor(state.__array__())
            state = state.unsqueeze(0)
        
        #print(f"state shape {state.shape} ")
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = self.online_network(state)


        action = self.env.action_space.noop()
        
        # select a random action wih probability eps
        if self.exploration :
                if random.random() <= self.epsilon:
                    #print("RANDOM ACTION")
                    action[self.action_codes[random.randint(0,self.nb_possible_action-1)]]=1
                    #print("action sampled automaticly ", action)
                    #print("its code is ",self.get_code_from_env_action(action))
                else:
                    #print(f'dqn choose : {self.action_codes[np.argmax(values.cpu().numpy())]} ')
                    action[self.action_codes[np.argmax(values.cpu().numpy())]]=1
        else:
            #print(f'dqn choose : {self.action_codes[np.argmax(values.cpu().numpy())]} ')
            action[self.action_codes[np.argmax(values.cpu().numpy())]]=1

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
            rewards = []
            for i_episode in range(NB_EPISODES_TRAIN):
                
                observationN = self.env.reset()
                cumul_reward = 0
                nb_actions = 0
                done = False
                for t in range(MAX_ACTIONS_PER_EPISODES):
                #while not done:
                    # if RENDRING_ENV : self.env.render()      
                    # Choix de l'action
                    action = self.choose_action(observationN)
                    action_code = self.get_code_from_env_action(action)
                    #print(f" action_code : {action_code}") 
                    observationF, reward, done, info = self.env.step(action)
                    
                    cumul_reward += reward
                    nb_actions+=1
                    self.memory.add(observationN,action_code,reward,observationF,done)

                    observationN=observationF
                    
                    if self.memory.__len__ () > BATCH_SIZE:
                        self.learn(self.optim,self.memory)
                    
                    
                    # exploration alpha greedy avec decay
                    if self.exploration :
                        self.update_epsilon()

                    if done:
                        break
                rewards.append(cumul_reward)
                list_reward[i_episode] = cumul_reward
                plot_evolution(rewards,'Train')
                print("Episode finished after {} timesteps cumul rewards {}".format(nb_actions,cumul_reward))
                #episode_durations.append(nb_actions)
                #print("CUMUL REWARDS FOR EPISODE ",i_episode,"is :",cumul_reward)           
            if SAVE_MODEL_PARAMS:
                torch.save(self.online_network.state_dict(), MODEL_PARAMS_PATH)


        if TEST:
            observationN = self.env.reset()
            episode_durations = []
            list_reward = {}
            # Pour ne plus explorer
            if RECORD_PERFS:
                video_path='demo-MineRl.mp4'
                video_recorder_object =video_recorder.VideoRecorder(self.env, path=video_path)

            self.exploration = False
            for i_episode in range(NB_EPISODES_TEST):
                observationN = self.env.reset()
                cumul_reward = 0
                nb_actions = 0
                done = False
                #for t in range(MAX_ACTIONS_PER_EPISODES):
                while not done:
                    #if RENDRING_ENV : self.env.render()  
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
            
            video_recorder_object.close()
            video_recorder_object.enabled = False
            self.env.close() 
            
            moy = np.mean(list(list_reward.values()))
            ecart_type= np.std(list(list_reward.values()))
            print(f" Moyenne +/- Ecart type  des récompenses sur les {NB_EPISODES_TEST} episodes")
            print(f" Moyennes : {moy}")
            print(f" Ecart type  : {ecart_type}")

    
    def learn(self, optim, memory ):
        self.compteur_train +=1 
        # Juste avec un seul model 
        resultats = self.memory.sample()
        # Pretraitement des etats 
        if not FRAME_STACKING:  
            future_states  = torch.cat([x.unsqueeze(0) for x in resultats[3]], 0)
            future_states = future_states.unsqueeze(1)

            states  = torch.cat([x.unsqueeze(0) for x in resultats[0]], 0)
            states = states.unsqueeze(1)
        else:
            future_states  = torch.cat([torch.FloatTensor(x.__array__()).unsqueeze(0) for x in resultats[3]], 0)
            #future_states = future_states.unsqueeze(1)

            states  = torch.cat([torch.FloatTensor(x.__array__()).unsqueeze(0) for x in resultats[0]], 0)
            #states = states.unsqueeze(1)
            #print(f" future state shape {future_states.shape}")
            #print(f" state shape {states.shape}")

        #print(f" future state shape {future_states.shape}")

        #target q values : 
        target_q_values = self.target_network(future_states)

        #MAX TARGET Q VALUE 
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = resultats[2] + self.gamma * max_target_q_values * (1 - resultats[4])
    
        #print(resultats[0][0].shape)
        
        #print(f" state shape {states.shape}")
        
        q_values = self.online_network(states)
        
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


ae = DQNAgent(env_name)
ae.run()


