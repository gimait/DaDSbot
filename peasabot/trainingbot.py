import random

from coderone.dungeon.agent import GameState, PlayerState
from peasabot.dqn_impl import GameQLearner

class agent():
    """ Agent bot."""
    def __init__(self):
        self.n = random.randint(0, 20)
        self.actions = ['', 'u', 'd', 'l', 'r', 'p']
        self.dqn = GameQLearner(2)

    def next_move(self, game_state: GameState, player_state: PlayerState):
        return random.choice(self.actions)
        return self.dqn.give_next_move(game_state, player_state)

    def reset(self):
        pass
