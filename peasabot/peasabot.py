"""
Our peasant bot.
"""
from coderone.dungeon.agent import GameState, PlayerState

from .consumer_bot import ConsumerBot
from .map_prep import DistanceMap, FreedomMap, BombMap


class Agent(ConsumerBot):
    """ Agent bot."""
    def __init__(self) -> None:
        self.actions = ['', 'u', 'd', 'l', 'r', 'p']
        self.map_representation = DistanceMap((12, 10))
        self.free_map = FreedomMap((12, 10))
        self.bomb_target_map = BombMap((12, 10))
        self.planned_actions = []
        self.full_map_prev = None
        self.substrategy = 1
        self.current_bombs = []

        # Attributes updated in the tick
        # self.cols #self.rows               Map Size

        # self.game_state                    Whole game_state
        # self.location                      Location of your player

        # self.ammo
        # self.bombs

        # self.bombs_in_range
        # self.surrounding_tiles
        # self.empty_tiles

    def next_move(self, game_state: GameState, player_state: PlayerState):
        self.update_state(game_state, player_state)
        self.map_representation.update(game_state, player_state.location, player_state.id)
        self.free_map.update(game_state, player_state.location, player_state.id)
        self.bomb_target_map.update(game_state, player_state.location, player_state.id)

        updated_map = (self.full_map_prev == self.print_map if self.full_map_prev is not None else True)
        # If the map is not updated dont change the plan (elif)
        if not self.planned_actions:
            self.planned_actions = self.next_move_killer()
        elif self.planned_actions and updated_map:
            self.planned_actions = self.next_move_killer()
        action = (self.planned_actions.pop() if self.planned_actions else '')

        # Update the map for checking if change in the next tick
        self.full_map_prev = self.print_map

        return action

    def reset(self):
        pass
