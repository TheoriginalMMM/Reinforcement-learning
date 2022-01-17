from environment.custom_handlers import *

from minerl.herobraine.hero.mc import MS_PER_STEP, INVERSE_KEYMAP
from minerl.herobraine.env_spec import EnvSpec
import minerl.herobraine.hero.handlers as handlers

import os
from abc import ABC


SIMPLE_KEYBOARD_ACTION = ['left', 'right', 'attack']


class Mineline(EnvSpec, ABC):
    def __init__(self, name, world, dense=False, resolution=(224, 224), max_episode_steps=300, *args, **kwargs):
        self.name = name
        self.world = world
        self.dense = dense
        self.resolution = resolution
        self.max_episode_steps = max_episode_steps
        super().__init__(name, max_episode_steps, *args, **kwargs)

    def is_from_folder(self, folder):
        return folder == 'custom'

    def create_observables(self):
        return [handlers.POVObservation(self.resolution)]

    def create_actionables(self):
        return [handlers.KeybasedCommandAction(k, v) for k, v in INVERSE_KEYMAP.items() if k in SIMPLE_KEYBOARD_ACTION]

    def create_rewardables(self):
        if self.dense:
            return [handlers.RewardForDistanceTraveledToCompassTarget(reward_per_block=5.0),
                FixedRewardForStructureCopying(5, 100),
                RewardForTimeTaken(-1)
            ]
        else:
            return [FixedRewardForStructureCopying(5, 100),
                RewardForTimeTaken(-1)
            ]

    def create_agent_start(self):
        return [handlers.SimpleInventoryAgentStart([dict(type='diamond_pickaxe', quantity='1')])]

    def create_agent_handlers(self):
        return []

    def create_server_world_generators(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'worlds', self.world)
        return [FixedFileWorldGenerator(path, destroy_after_use=True, force_reset=True)]

    def create_server_quit_producers(self):
        return [handlers.ServerQuitFromTimeUp(self.max_episode_steps * MS_PER_STEP),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self):
        return [MinelineDrawBlockDecorator(),
            MinelineBuildBattleDecorator(),
            MinelineNavigationDecorator()
        ]

    def create_server_initial_conditions(self):
        return [handlers.TimeInitialCondition(allow_passage_of_time=False, start_time=6000),
            handlers.WeatherInitialCondition('clear'),
            handlers.SpawningInitialCondition('false')
        ]

    def get_docstring(self):
        return """-"""

    def determine_success_from_rewards(self, rewards):
        return False

    def create_monitors(self):
        return []
