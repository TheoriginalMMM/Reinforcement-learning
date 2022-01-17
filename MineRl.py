
import matplotlib 
import matplotlib.pyplot as plt
from collections import namedtuple, deque

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
with open(HYPER_PARAMS_PATH, "w") as fichier:
    for i in HPARAMS.keys():
	    fichier.write(f"{i} : {HPARAMS[i]} \n")


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

env = gym.make('Mineline-v0')
env.reset()
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = gym.wrappers.FrameStack(env, 4)

#env = FrameStack(env, num_stack=4)

obs = env.reset()
print(obs)

for i in range(10):
    # Initialitation of the action
    action = env.action_space.noop()
    # Hwo to choos a action 
    #action['right'] = 0
    #action['left'] = 1
    # Hwo to take a random action 
    action = env.action_space.sample
    obs, reward, done, _ = env.step(action)
    print("NEW FRAME")
    print("Avec PRE TRAITEMENT ")
    print(type(obs))
    print(obs.shape)

#########################################################################


