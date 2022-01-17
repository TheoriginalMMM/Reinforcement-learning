import jinja2
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.handlers import RewardHandler

import random

class FixedNavigationDecorator(Handler):
    """ Specifies the navigate goal.
    """

    def to_string(self) -> str:
        return "navigation_decorator"

    def xml_template(self) -> str:
        return str(
            """<NavigationDecorator>
                <randomPlacementProperties>
                    <origin x="{{x}}" y="{{y}}" z="{{z}}"></origin>
                    <maxRandomizedRadius>1</maxRandomizedRadius>
                    <minRandomizedRadius>0</minRandomizedRadius>
                    <maxRadius>1</maxRadius>
                    <minRadius>0</minRadius>
                    <block>{{block}}</block>
                    <placement>sphere</placement>
                </randomPlacementProperties>
                <minRandomizedDistance>0</minRandomizedDistance>
                <maxRandomizedDistance>1</maxRandomizedDistance>
                <randomizeCompassLocation>false</randomizeCompassLocation>
            </NavigationDecorator>
            """
        )

    def __init__(self, x, y, z, block='diamond_block'):
        """Initialize navigation decorator

            :param x: Position of the goal on the x axis
            :param y: Position of the goal on the y axis
            :param z: Position of the goal on the z axis
            :param block: Type of block to appear.
        """
        self.x = x
        self.y = y
        self.z = z
        self.block = block



class FixedFileWorldGenerator(Handler):
    """Generates a world from a file."""

    def to_string(self) -> str:
        return "file_world_generator"

    def xml_template(self) -> str:
        return str(
            """<FileWorldGenerator
                destroyAfterUse = "{{destroy_after_use | string | lower}}"
                forceReset = "{{force_reset | string | lower}}"
                src = "{{filename}}" />
            """
        )

    def __init__(self, filename: str, destroy_after_use: bool = True, force_reset: bool = True):
        self.filename = filename
        self.destroy_after_use = destroy_after_use
        self.force_reset = force_reset



class RewardForTimeTaken(RewardHandler):
    def to_string(self) -> str:
        return "reward_for_time_taken"

    def xml_template(self) -> str:
        return str(
            """<RewardForTimeTaken
            dimension="0"
            initialReward="0"
            delta="{{delta}}"
            density="PER_TICK"/>"""
        )

    def __init__(self, delta):
        self.delta = delta

    def from_universal(self, obs):
        return self.delta



class FixedRewardForMissionEnd(RewardHandler):
    def to_string(self) -> str:
        return "reward_for_mission_end"

    def xml_template(self) -> str:
        return str(
            """<RewardForMissionEnd rewardForDeath="{{ rewardDeath }}">
                    <Reward description="{{ description }}" reward="{{ reward }}" />
                </RewardForMissionEnd>"""
        )

    def __init__(self, reward: int = 0, description: str = "out_of_time", rewardDeath: int = 0):
        """Creates a reward which is awarded when a mission ends."""
        super().__init__()
        self.reward = reward
        self.description = description
        self.rewardDeath = rewardDeath

    def from_universal(self, obs):
        return 0



class FixedRewardForStructureCopying(RewardHandler):  
    def to_string(self) -> str:
        return "reward_for_structure_copying"

    def xml_template(self) -> str:
        return str(
            """<RewardForStructureCopying
                rewardScale="{{rewardScale}}"
                rewardForCompletion="{{rewardForCompletion}}">
                <RewardDensity>{{rewardDensity}}</RewardDensity>
                <AddQuitProducer description="structure_copied"/>
                </RewardForStructureCopying>"""
        )

    def __init__(self, rewardScale: int, rewardForCompletion: int, rewardDensity: str = "PER_BLOCK"):
        super().__init__()
        self.rewardScale = rewardScale
        self.rewardForCompletion = rewardForCompletion
        self.rewardDensity = rewardDensity

    def from_universal(self, obs):
        return 0


SEED = 0
SEED_USED = 0
class MinelineBuildBattleDecorator(Handler):
    def to_string(self) -> str:
        return "build_battle_decorator"

    def xml_template(self) -> str:
        global SEED
        global SEED_USED
        random.seed(SEED)
        randomX = random.randint(-3, 3)

        SEED_USED += 1
        if SEED_USED == 3:
            SEED_USED = 0
            SEED += 1

        return str(
            """<BuildBattleDecorator>
                    <GoalStructureBounds>
                        <min x="{x}" y="102" z="6"/>
                        <max x="{x}" y="102" z="6"/>
                    </GoalStructureBounds>
                    <PlayerStructureBounds>
                        <min x="{x}" y="102" z="2"/>
                        <max x="{x}" y="102" z="2"/>
                    </PlayerStructureBounds>
                </BuildBattleDecorator>""".format(x=randomX)
        )

    def __init__(self):
        pass



class MinelineDrawBlockDecorator(Handler):
    def to_string(self) -> str:
        return "draw_block_decorator"

    def xml_template(self) -> str:
        global SEED
        global SEED_USED
        random.seed(SEED)
        randomX = random.randint(-3, 3)

        SEED_USED += 1
        if SEED_USED == 3:
            SEED_USED = 0
            SEED += 1

        return str(
            """<DrawingDecorator>
                    <DrawBlock type="diamond_block" x="{x}" y="102" z="2"/>
                    <DrawBlock type="air" x="{x}" y="102" z="6"/>
                </DrawingDecorator>""".format(x=randomX)
        )

    def __init__(self):
        pass



class MinelineNavigationDecorator(Handler):
    def to_string(self) -> str:
        return "navigation_decorator"

    def xml_template(self) -> str:
        global SEED
        global SEED_USED
        random.seed(SEED)
        randomX = random.randint(-3, 3)

        SEED_USED += 1
        if SEED_USED == 3:
            SEED_USED = 0
            SEED += 1

        return str(
            """<NavigationDecorator>
                <randomPlacementProperties>
                    <origin x="{x}" y="102" z="2"></origin>
                    <maxRandomizedRadius>1</maxRandomizedRadius>
                    <minRandomizedRadius>0</minRandomizedRadius>
                    <maxRadius>1</maxRadius>
                    <minRadius>0</minRadius>
                    <block>diamond_block</block>
                    <placement>sphere</placement>
                </randomPlacementProperties>
                <minRandomizedDistance>0</minRandomizedDistance>
                <maxRandomizedDistance>1</maxRandomizedDistance>
                <randomizeCompassLocation>false</randomizeCompassLocation>
            </NavigationDecorator>""".format(x=randomX)
        )

    def __init__(self):
        pass



FLAT_JUMPS = {
    'easy':     [[-1, 0, 1], [0, 0, 1], [1, 0, 1]],
    'normal':   [[-2, 0, 2], [-1, 0, 2], [0, 0, 2], [1, 0, 2], [2, 0, 2], [-2, 0, 3], [-1, 0, 3], [0, 0, 3], [1, 0, 3], [2, 0, 3]],
    'hard':     [[-3, 0, 3], [3, 0, 3], [-2, 0, 4], [-1, 0, 4], [0, 0, 4], [1, 0, 4], [2, 0, 4]],
    'expert':   [[-3, 0, 4], [3, 0, 4], [-1, 0, 5], [0, 0, 5], [1, 0, 5]]
}

UP_JUMPS = {
    'easy':     [[-1, 1, 1], [0, 1, 1], [1, 1, 1]],
    'normal':   [[-2, 1, 2], [-1, 1, 2], [0, 1, 2], [1, 1, 2], [2, 1, 2]],
    'hard':     [[-2, 1, 3], [-1, 1, 3], [0, 1, 3], [1, 1, 3], [2, 1, 3]],
    'expert':   [[-3, 1, 3], [3, 1, 3], [-1, 1, 4], [0, 1, 4], [1, 1, 4]]
}

DOWN_JUMPS = {
    'easy':     [[-1, -1, 1], [0, -1, 1], [1, -1, 1], [-2, -1, 2], [-1, -1, 2], [0, -1, 2], [1, -1, 2], [2, -1, 2]],
    'normal':   [[-3, -1, 3], [-2, -1, 3], [-1, -1, 3], [0, -1, 3], [1, -1, 3], [2, -1, 3], [3, -1, 3]],
    'hard':     [[-3, -1, 4], [-2, -1, 4], [-1, -1, 4], [0, -1, 4], [1, -1, 4], [2, -1, 4], [3, -1, 4], [-2, -1, 5], [-1, -1, 5], [0, -1, 5], [1, -1, 5], [2, -1, 5]],
    'expert':   [[-4, -1, 4], [4, -1, 4], [-3, -1, 5], [3, -1, 5]]
}

DOWNDOWN_JUMPS = {
    'easy':     [[-1, -2, 1], [0, -2, 1], [1, -2, 1], [-2, -2, 2], [-1, -2, 2], [0, -2, 2], [1, -2, 2], [2, -2, 2]],
    'normal':   [[-3, -2, 3], [-2, -2, 3], [-1, -2, 3], [0, -2, 3], [1, -2, 3], [2, -2, 3], [3, -2, 3]],
    'hard':     [[-3, -2, 4], [-2, -2, 4], [-1, -2, 4], [0, -2, 4], [1, -2, 4], [2, -2, 4], [3, -2, 4], [-3, -2, 5], [-2, -2, 5], [-1, -2, 5], [0, -2, 5], [1, -2, 5], [2, -2, 5], [3, -2, 5]],
    'expert':   [[-4, -2, 4], [4, -2, 4]]
}

class ParkourInfinitDrawBlockDecorator(Handler):
    def to_string(self) -> str:
        return "draw_block_decorator"

    def xml_template(self) -> str:
        block = [0, 100, 0]
        string = '<DrawingDecorator>'

        for i in range(1000):
            if block[1] < 102:
                jumpType = random.choice(['FLAT', 'FLAT', 'UP'])
            elif block[1] > 108:
                jumpType = random.choice(['FLAT', 'FLAT', 'FLAT', 'FLAT', 'DOWN', 'DOWNDOWN'])
            else:
                jumpType = random.choice(['FLAT', 'FLAT', 'FLAT', 'FLAT', 'FLAT', 'UP', 'UP', 'UP', 'DOWN', 'DOWNDOWN'])

            if jumpType == 'UP':
                jumpList = UP_JUMPS
            elif jumpType == 'DOWN':
                jumpList = DOWN_JUMPS
            elif jumpType == 'DOWNDOWN':
                jumpList = DOWNDOWN_JUMPS
            else:
                jumpList = FLAT_JUMPS

            if i < 30:
                jumpDifficulty = random.choice(['easy'])
            elif i < 100:
                jumpDifficulty = random.choice(['easy', 'normal'])
            elif i < 200:
                jumpDifficulty = random.choice(['normal', 'hard'])
            elif i < 300:
                jumpDifficulty = random.choice(['normal', 'hard', 'expert'])
            elif i < 500:
                jumpDifficulty = random.choice(['hard', 'expert'])
            else:
                jumpDifficulty = random.choice(['expert'])
            
            jump = random.choice(jumpList[jumpDifficulty])
            block = [sum(x) for x in zip(block, jump)]

            string += '<DrawBlock type="stained_hardened_clay" colour="{}" x="{}" y="{}" z="{}"/>'.format('GREEN' if i%2 == 0 else 'RED', *block)
        
        string += '</DrawingDecorator>'

        return str(string)

    def __init__(self):
        pass



class DigdownDrawBlockDecorator(Handler):
    def to_string(self) -> str:
        return "draw_block_decorator"

    def xml_template(self) -> str:
        string = '<DrawingDecorator>'

        for i in range(11, 101):
            rng = random.randint(0, 2)
            
            if rng == 1:
                string += '<DrawBlock type="dirt" x="0" y="{}" z="0"/>'.format(i)
            elif rng == 2:
                string += '<DrawBlock type="log" x="0" y="{}" z="0"/>'.format(i)
            else:
                string += '<DrawBlock type="stone" x="0" y="{}" z="0"/>'.format(i)
        
        string += '</DrawingDecorator>'

        return str(string)

    def __init__(self):
        pass