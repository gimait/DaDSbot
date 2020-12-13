"""
Consumer bot
"""

import random
from typing import List, Optional, Tuple

from coderone.dungeon.agent import GameState, PlayerState

import numpy as np

from .map_prep import BombMap, DistanceMap, FreedomMap
from .timers import TimeBomb
from .utilities import get_opponents

TIMEOUT = 20  # Maximum of positions to calculate in the planning
MAX_BOMB = 5  # Dont pick up more
MIN_BOMB = 1  # Dont place bomb

class ConsumerBot:
    __slots__ = [
        "size",
        "map_representation",
        "free_map",
        "bomb_target_map",
        "__dict__"
    ]

    def __init__(self, size) -> None:
        self.actions = ['', 'u', 'd', 'l', 'r', 'p']
        self.size = size

        self.map_representation = DistanceMap(self.size)
        self.free_map = FreedomMap(self.size)
        self.bomb_target_map = BombMap(self.size)
        self.previous_plan = None

        self.planned_actions = []
        self.full_map_prev = None
        self.substrategy = 1
        self.current_bombs = []

        # Attributes updated in the tick
        self.game_state = None  # Whole game_state
        self.location = None  # Location of your player
        self.ammo = None
        self.bombs = None

        self.bombs_in_range = None
        self.surrounding_tiles = None
        self.empty_tiles = None

    def update_state(self, game_state: GameState, player_state: PlayerState) -> None:
        ########################
        # ##   AGENT STATE  ## #
        ########################

        # store some information about the environment
        # game map is represented in the form (x,y)
        self.cols = game_state.size[0]
        self.rows = game_state.size[1]

        # for us to refer to later
        self.game_state = game_state
        self.player_state = player_state

        self.location = player_state.location
        self.ammo = player_state.ammo
        self.hp = player_state.hp
        self.reward = player_state.reward
        self.power = player_state.power

        # Check new bombs and update timers
        if game_state.bombs:
            if not self.current_bombs:
                for bomb in game_state.bombs:
                    self.current_bombs.append(TimeBomb(game_state.tick_number, bomb))
            else:
                # Update new bombs
                stable_bomb_pos = []
                stable_bomb_timer = []
                for cb in self.current_bombs:
                    if cb.position in game_state.bombs:
                        stable_bomb_pos.append(cb.position)
                        stable_bomb_timer.append(cb)

                new_bombs = [TimeBomb(game_state.tick_number, bomb) for bomb in game_state.bombs
                             if bomb not in stable_bomb_pos]
                stable_bombs = [bomb for bomb in self.current_bombs if bomb in stable_bomb_timer]
                self.current_bombs = new_bombs + stable_bombs
        else:
            self.current_bombs = []

    def get_cross_tiles(self, tile: Tuple[int, int]) -> List[Tuple[int, int]]:
        """ From a given tile outputs the neighbour tiles in the 4 directions """
        new_tile_r = (tile[0] - 1, tile[1])
        new_tile_u = (tile[0], tile[1] - 1)
        new_tile_l = (tile[0] + 1, tile[1])
        new_tile_d = (tile[0], tile[1] + 1)
        list_tiles = [new_tile_r, new_tile_u, new_tile_l, new_tile_d]
        return list_tiles

    def get_closest_ammo(self):
        distance = 999  # here check with the value of the no possible from the value_at_point
        tile = None
        for i, ammo_tile in enumerate(self.game_state.ammo):
            d2p_ammo = self.map_representation.value_at_point(ammo_tile)
            if d2p_ammo != 0 and d2p_ammo < distance:
                distance = d2p_ammo
                tile = ammo_tile
        return tile

    def get_best_point_for_bomb(self):
        optimal_points = np.multiply(self.map_representation._map, self.bomb_target_map._map)
        return np.unravel_index(optimal_points.argmax(), optimal_points.shape)

    def get_freedom_tiles(self):
        freedom_tiles = np.multiply(self.map_representation._map, self.bomb_target_map._map)
        return np.unravel_index(freedom_tiles.argmax(), freedom_tiles.shape)

    def evaluate_bomb(self, tiles_list):
        # Input a list of tiles to evaluate Outputs the best tile to place a bomb
        # First it only considers the closest distance if any better strategy is expected here
        weight = np.array([])
        tlist = ([t for t in tiles_list if self.game_state.is_in_bounds(t)])
        tnplist = np.array(tlist)
        if not tlist:
            return([])
        for tiles in tlist:
            self.map_representation.value_at_point(tiles)
            weight = np.append(weight, self.map_representation.value_at_point(tiles))
        index = np.where(weight == np.amin(weight))
        tile = (tnplist[index].tolist())[-1]
        return tile

    def path_to_safest_area(self, danger_zone: Optional[List[Tuple[int, int]]]):
        # take out unsafe tiles from free_map:
        for tile in danger_zone:
            if tile[0] >= 0 and tile[1] >= 0 and tile[0] < self.free_map.size[0] and tile[1] < self.free_map.size[1]:
                self.free_map._map[tile] = 0
        safety_map = np.multiply(self.free_map._map, self.map_representation._map)

        safest_tile = np.unravel_index(safety_map.argmax(), safety_map.shape)

        return self.plan_to_tile(safest_tile)

    def plan_to_tile(self, goal_tile: Tuple[int, int]) -> Tuple[List, bool]:
        plan = []
        tile = tuple(goal_tile)

        if self.map_representation.value_at_point(tile) == 0:
            return plan, False

        timeout = TIMEOUT
        ite = 0
        while tile != self.location and ite != timeout:
            moves = np.array([])
            weight = np.array([])
            # Movements
            list_tiles = self.get_cross_tiles(tile)
            (new_tile_r, new_tile_u, new_tile_l, new_tile_d) = (list_tiles[0], list_tiles[1],
                                                                list_tiles[2], list_tiles[3])
            if self.game_state.is_in_bounds(new_tile_r) and self.map_representation.value_at_point(new_tile_r) > 0:
                moves = np.append(moves, 'r')
                weight = np.append(weight, self.map_representation.value_at_point(new_tile_r))
            # Up
            if self.game_state.is_in_bounds(new_tile_u) and self.map_representation.value_at_point(new_tile_u) > 0:
                moves = np.append(moves, 'u')
                weight = np.append(weight, self.map_representation.value_at_point(new_tile_u))
            # Left
            if self.game_state.is_in_bounds(new_tile_l) and self.map_representation.value_at_point(new_tile_l) > 0:
                moves = np.append(moves, 'l')
                weight = np.append(weight, self.map_representation.value_at_point(new_tile_l))
            # Down
            if self.game_state.is_in_bounds(new_tile_d) and self.map_representation.value_at_point(new_tile_d) > 0:
                moves = np.append(moves, 'd')
                weight = np.append(weight, self.map_representation.value_at_point(new_tile_d))

            # minimum value
            if weight.size == 0:
                return([''], False)

            index = np.where(weight == np.amin(weight))
            movement = (moves[index].tolist())[-1]
            plan.append(movement)

            # Update tile and iteration
            ite = ite + 1
            if movement == 'r':
                tile = new_tile_r
            elif movement == 'u':
                tile = new_tile_u
            elif movement == 'l':
                tile = new_tile_l
            elif movement == 'd':
                tile = new_tile_d
            else:
                break
        return plan, True

    def is_in_danger(self):
        bombs_about_to_explode = []
        danger_zone = []
        for b in self.current_bombs:
            if b.time_to_explode(self.game_state.tick_number) < 10:
                bombs_about_to_explode.append(b.position)

        for b in bombs_about_to_explode:
            if abs(b[0] - self.location[0]) <= 2 or abs(b[1] - self.location[1]) <= 2:
                danger_zone = danger_zone + [b, (b[0] - 2, b[1]), (b[0] - 1, b[1]), (b[0] + 2, b[1]), (b[0] + 1, b[1]),
                               (b[0], b[1] - 2), (b[0], b[1] - 1), (b[0], b[1] + 2), (b[0], b[1] + 1)]
        status = (True if danger_zone else False)
        return danger_zone, status

    def is_ammo_avail(self):
        ammo_tile = self.get_closest_ammo()
        if ammo_tile is not None and self.map_representation.value_at_point(ammo_tile) > 0:
            status = True
        else:
            status = False
        return ammo_tile, status

    def is_killing_an_option(self):
        tiles_list = []
        for opponent_tile in get_opponents(self.player_state.id, self.game_state._players):
            t = self.get_cross_tiles(opponent_tile)
            tiles_list = tiles_list + t
        bomb_tile = self.evaluate_bomb(tiles_list)
        status = (True if bomb_tile else False)
        return bomb_tile, status

    def next_move_killer(self):
        # Agent possibilities
        danger_zone, danger_status = self.is_in_danger()
        ammo_tile, ammo_status = self.is_ammo_avail()
        kill_tiles, kill_status = self.is_killing_an_option()

        # 1 Avoid Danger
        if danger_status:
            plan, _ = self.path_to_safest_area(danger_zone)
        # 2 Pick up ammo if less than MAX
        elif ammo_status and self.ammo < MAX_BOMB:
            plan, _ = self.plan_to_tile(ammo_tile)
        # 3 Plan for killing, finish it if started
        elif self.previous_plan == "kill" or kill_status:
            plan, connected = self.plan_to_tile(kill_tiles)
            self.previous_plan = (None if not plan else "kill")
            plan.insert(0, 'p')
        # 4 Place a bomb in a good place if you have bombs
        elif self.previous_plan == "loot" or self.ammo > MIN_BOMB:
            best_point_for_bomb = self.get_best_point_for_bomb()
            plan, _ = self.plan_to_tile(best_point_for_bomb)
            self.previous_plan = (None if not plan else "loot")
            plan.insert(0, 'p')
        # Last Find a good place to wait
        else:
            free_tile = self.get_freedom_tiles()
            plan, _ = self.plan_to_tile(free_tile)

        return plan



    def next_move_bombAvoider(self, game_state, player_state):
        """ Call each time the agent is required to choose an action """
        ########################
        # ##   VARIABLES    ## #
        ########################

        # if I'm on a bomb, I should probably move
        if game_state.entity_at(self.location) == 'b':

            print("I'm on a bomb. I'm going to move.")

            if self.empty_tiles:
                # choose a random free tile to move to
                random_tile = random.choice(self.empty_tiles)
                action = self.move_to_tile(self.location, random_tile)
            else:
                # if there isn't a free spot to move to, we're probably stuck here
                action = ''

        # if we're near a bomb, we should also probably move
        elif self.bombs_in_range:

            print("I'm fleeing.")

            if self.empty_tiles:

                # get the safest tile for us to move to
                safest_tile = self.get_safest_tile(self.empty_tiles, self.bombs_in_range)

                action = self.move_to_tile(self.location, safest_tile)

            else:
                action = random.choice(self.actions)

        # if there are no bombs in range
        else:

            print("I'm placing a bomb")

            # but first, let's check if we have any ammo
            if self.ammo > 0:
                # we've got ammo, let's place a bomb
                action = 'p'
            else:
                # no ammo, we'll make random moves until we have ammo
                action = random.choice(self.actions)

        return action

    @staticmethod
    def print_map(game_state):
        """ Print the game map as an array using numpy """
        cols = game_state.size[0]
        rows = game_state.size[1]

        game_map = []
        for x in range(cols):
            for y in range(rows):
                game_map.append(( game_state.entity_at((x, y)) if game_state.entity_at((x, y)) is not None else 'f'))

        return game_map
