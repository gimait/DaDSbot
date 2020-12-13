"""
Our peasant bot.
"""
from coderone.dungeon.agent import GameState, PlayerState

from .consumer_bot import ConsumerBot


class Agent(ConsumerBot):
    """ Agent bot."""
    def __init__(self):
        super().__init__((12, 10))

    def next_move(self, game_state: GameState, player_state: PlayerState):
        self.update_state(game_state, player_state)
        self.map_representation.update(game_state, player_state.location, player_state.id)
        self.free_map.update(game_state, player_state.location, player_state.id)
        self.bomb_target_map.update(game_state, player_state.location, player_state.id)

        updated_map = (self.full_map_prev == self.print_map(game_state) if self.full_map_prev is not None else True)
        # If the map is not updated dont change the plan (elif)
        if not self.planned_actions:
            self.planned_actions = self.next_move_killer()
        elif updated_map:
            self.planned_actions = self.next_move_killer()
        action = (self.planned_actions.pop() if self.planned_actions else '')

        # Update the map for checking if change in the next tick
        self.full_map_prev = self.print_map(game_state)

        return action

    def reset(self):
        pass
