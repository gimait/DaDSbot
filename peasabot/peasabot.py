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

        full_map = self.print_map(game_state)
        updated_map = (self.full_map_prev == full_map if self.full_map_prev is not None else True)
        # Yes I debug with a
        print(updated_map)
        if not self.planned_actions:
            self.planned_actions = self.next_move_killer()
        elif updated_map:
            self.planned_actions = self.next_move_killer()
        else:  # Map is not updated and there are planned actions
            pass
        action = (self.planned_actions.pop() if self.planned_actions else '')

        self.full_map_prev = full_map  # Update the map for checking if change in the next tick
        self.last_move = action

        return action

    def reset(self):
        pass
