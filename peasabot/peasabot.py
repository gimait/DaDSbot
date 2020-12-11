"""

    Our peasant bot.

"""
from coderone.dungeon.agent import GameState, PlayerState


class agent:
    def __init__(self) -> None:
        self.actions = ['', 'u', 'd', 'l', 'r', 'p']
        pass

    def next_move(self, game_state: GameState, player_state: PlayerState):
        return self.actions[0]

    def reset(self):
        pass
