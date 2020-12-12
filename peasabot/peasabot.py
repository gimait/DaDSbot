"""
Our peasant bot.
"""
from coderone.dungeon.agent import GameState, PlayerState
from consumer_bot import Consumer_bot

class agent(Consumer_bot):
    """ Agent bot."""
    def __init__(self) -> None:
        self.actions = ['', 'u', 'd', 'l', 'r', 'p']
        pass

    def next_move(self, game_state: GameState, player_state: PlayerState):
        return self.actions[2]

    def reset(self):
        pass
