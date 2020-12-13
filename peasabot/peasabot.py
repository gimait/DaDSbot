"""
Our peasant bot.
"""
from coderone.dungeon.agent import GameState, PlayerState

from .map_prep import DistanceMap, BombMap, FreedomMap


class agent():
    """ Agent bot."""
    def __init__(self) -> None:
        self.actions = ['', 'u', 'd', 'l', 'r', 'p']
        gamesize = (12, 10)
        self.map_representations = [DistanceMap(gamesize), BombMap(gamesize), FreedomMap(gamesize)]

    def next_move(self, game_state: GameState, player_state: PlayerState):
        for rep in self.map_representations:
            rep.update(game_state, player_state.location, player_state.id)

        # do stuff here to plan your next action
        return 0

    def reset(self):
        pass
