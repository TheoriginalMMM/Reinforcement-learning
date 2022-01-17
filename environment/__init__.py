from environment.parkour_specs import Parkour
from environment.parkour_infinit_specs import ParkourInfinit
from environment.mineline_specs import Mineline
from environment.maze_specs import Maze
from environment.digdown_specs import Digdown
from environment.digforward_specs import Digforward
from environment.fight_specs import Fight

import gym

maps = []


maps.append(Mineline('Mineline-v0', 'Mineline', dense=False))
maps.append(Mineline('MinelineDense-v0', 'Mineline', dense=True))

maps.append(Parkour([0, 100, 56], 'Parkour-v0', 'Parkour'))

maps.append(ParkourInfinit([0, 105, 1000000], 'ParkourInfinit-v0', 'ParkourInfinitQuick'))

maps.append(Maze([2, 100, 14], 'Maze-v0', 'MazeHelp', dense=False))
maps.append(Maze([2, 100, 14], 'MazeDense-v0', 'MazeHelp', dense=True))
maps.append(Maze([2, 100, 14], 'Maze-v1', 'Maze', dense=False))

maps.append(Digdown([0, 10, 0], 'Digdown-v0', 'Digdown'))

maps.append(Digforward([0, 100, 43], 'Digforward-v0', 'Digforward'))

maps.append(Fight('Fight-v0', 'Arena'))


for map in maps:
    if map.name not in gym.envs.registry.env_specs:
        map.register()
