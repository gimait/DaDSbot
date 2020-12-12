"""
Our peasant bot.
"""
from coderone.dungeon.agent import GameState, PlayerState

from .map_prep import DistanceMap


class agent():
    """ Agent bot."""
    def __init__(self) -> None:
        self.actions = ['', 'u', 'd', 'l', 'r', 'p']
        self.map_representation = DistanceMap((9, 11))


        pass

    def next_move(self, game_state: GameState, player_state: PlayerState):
        self.map_representation.update(game_state, player_state)

        if self.planned_actions:
            action = self.planned_actions.pop()
        else:
            self.next_move_()
        # do stuff here to plan your next action

        return action

    def reset(self):
        pass
