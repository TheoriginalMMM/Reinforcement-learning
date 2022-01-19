
import matplotlib 
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from gym.wrappers.frame_stack import FrameStack

import gym
from gym.spaces import Box




import environment

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import transforms as T


matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#PARAMETERS ET CONFIGURATIONS

TRAIN = True
TEST = True
SAVE_MODEL_PARAMS = True
LOAD_MODEL_PARAMS = True

MODEL_PARAMS_PATH = "CartPole.data"
HYPER_PARAMS_PATH = "Hyper_Params.txt"
#400
NB_EPISODES_TRAIN = 500
NB_EPISODES_TEST = 200
START_TRAIN = 1000

RECORD_PERFS = False
RENDRING_ENV = False

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

# Sauvgarder la liste des paramtres 
# with open(HYPER_PARAMS_PATH, "w") as fichier:
#     for i in HPARAMS.keys():
# 	    fichier.write(f"{i} : {HPARAMS[i]} \n")


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
        actions = torch.as_tensor(np.asarray([e.action for e in experiences if e is not None]), dtype=torch.int64).unsqueeze(-1)
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
        
# Testons nos wrapper de pré traitement
memory = ReplayBuffer(20,2,0)

env = gym.make('Mineline-v0')
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
#env = FrameStack(env,4)
print("SA PASSE !")


obs_i = env.reset()
print(obs_i)

for i in range(4):
    # Initialitation of the action
    print(env.action_space)
    action = env.action_space.noop()
    # Hwo to choos a action 
    #action['right'] = 0
    #action['left'] = 1
    # Hwo to take a random action 
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    action = list(action.values())
    memory.add(obs_i,action,reward,obs,done)
    obs_i = obs
    print(env.observation_space)
    print("NEW FRAME")
    print("Avec PRE TRAITEMENT ")
    print(type(obs))
    print(obs.shape)
    if done:
        break
   
batch = memory.sample()
print(batch[0])
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

        self.conv_1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_1 = nn.Linear(8960, 512)
        self.fc_2 = nn.Linear(512, action_dim)

    def forward(self, inp):
        inp = inp.view((1, 3, 210, 160))
        x1 = F.relu(self.conv_1(inp))
        x1 = F.relu(self.conv_2(x1))
        x1 = F.relu(self.conv_3(x1))
        x1 = torch.flatten(x1, 1)
        x1 = F.leaky_relu(self.fc_1(x1))
        x1 = self.fc_2(x1)

        return x1



#########################################################################


