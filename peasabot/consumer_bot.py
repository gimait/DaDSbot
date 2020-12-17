"""
Consumer bot
"""

import random
from typing import List, Optional, Tuple

from coderone.dungeon.agent import GameState, PlayerState

import numpy as np

from .map_prep import BombAreaMap, DistanceMap, FreedomMap, TargetMap
from .utilities import get_opponents, OreCounter


class ConsumerBot:
    __slots__ = [
        "size",
        "map_representation",
        "free_map",
        "bomb_target_map",
        "__dict__"
    ]

    def __init__(self, size, bomb_tick_threshold) -> None:
        self.actions = ['', 'u', 'd', 'l', 'r', 'p']
        self.size = size

        self.map_representation = DistanceMap(self.size)
        self.emergency_map = DistanceMap(self.size)
        self.free_map = FreedomMap(self.size)
        self.bomb_target_map = TargetMap(self.size)
        self.bomb_management_map = BombAreaMap(self.size, danger_thresh=bomb_tick_threshold)
        self.previous_plan = None

        self.planned_actions = []
        self.full_map_prev = None
        self.substrategy = 1
        self.ore_counter = None

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

        self.opponent_tile = get_opponents(self.player_state.id, self.game_state._players)[0]

        # Check new bombs and update timers
        self.bomb_management_map.update(game_state, player_state.location)
        self.free_map.update(game_state, player_state.location, player_state.id)
        self.bomb_target_map.update(game_state, player_state.location, player_state.id)
        self.map_representation.update(game_state, player_state.location, player_state.id, mask=self.bomb_management_map.danger_zone)
        self.emergency_map.update(game_state, player_state.location, player_state.id)

        # Initialize and update ores
        current_ores = game_state.ore_blocks
        if self.ore_counter is None:
            self.ore_counter = [OreCounter(pos) for pos in current_ores]

        i = 0
        while i < len(self.ore_counter):
            if self.ore_counter[i] not in current_ores:
                del self.ore_counter[i]
            else:
                for bomb in self.bomb_management_map.bombs:
                    if not bomb.counted and bomb._map[self.ore_counter[i].position]:
                        self.ore_counter[i].got_hit()
                        bomb.counted = True
                i += 1

    def get_closest_item(self, item):
        distance = 999  # here check with the value of the no possible from the value_at_point
        tile = None
        for i, item_tile in enumerate(item):
            d2p_ammo = self.map_representation.value_at_point(item_tile)
            if d2p_ammo != 0 and d2p_ammo < distance:
                distance = d2p_ammo
                tile = item_tile
        return tile

    def get_best_point_for_bomb(self):
        optimal_points = np.multiply(self.free_map._map,
                                      np.multiply(self.map_representation.distance_penalty_map,
                                                  self.bomb_target_map._map))
        return np.unravel_index(optimal_points.argmax(), optimal_points.shape)

    def get_freedom_tiles(self):
        freedom_tiles = np.multiply(self.map_representation.distance_penalty_map, self.free_map._map)
        return np.unravel_index(freedom_tiles.argmax(), freedom_tiles.shape)

    def evaluate_bomb(self, tiles_list):
        # Input a list of tiles to evaluate Outputs the best tile to place a bomb
        # First it only considers the closest distance if any better strategy is expected here
        tlist = ([t for t in tiles_list if self.game_state.is_in_bounds(t)])
        if not tlist:
            return([])
        best_tile = ()
        best_weight = 0
        for tile in tlist:
            tile_value = self.map_representation.value_at_point(tile)
            if tile_value > best_weight:
                best_tile = tile
        return best_tile

    def path_to_freest_area(self, danger_zone: Optional[BombAreaMap] = None):
        # take out unsafe tiles from free_map. Chooses the most free accesible area
        if danger_zone is not None:
            safety_map = np.multiply(danger_zone,
                                     np.multiply(self.free_map._map,
                                                 self.map_representation.accessible_area_mask))
        else:
            safety_map = np.multiply(self.free_map._map, self.map_representation._map)
        safest_tile = np.unravel_index(safety_map.argmax(), safety_map.shape)

        return self.plan_to_tile(safest_tile)

    def plan_to_safest_area(self, danger_zone: Optional[BombAreaMap] = None):
        # Map without bombs considered to escape asap but consider it for choosing the tile
        if danger_zone is not None:
            safety_map = np.multiply(danger_zone,
                                     np.multiply(self.free_map._map,
                                                 self.map_representation.distance_penalty_map))
        else:
            safety_map = np.multiply(self.free_map._map, self.map_representation._map)
        safest_tile = np.unravel_index(safety_map.argmax(), safety_map.shape)

        tiles, plan, connected = self.greedy_search(safest_tile, self.emergency_map)
        if not connected:
            return plan, connected
        return plan, connected

    def plan_to_tile(self, goal_tile: Tuple[int, int]) -> Tuple[List, bool]:
        tiles, plan, connected = self.greedy_search(goal_tile, self.map_representation)
        if not connected:
            return plan, connected
        plan_w_bomb_breaks = []
        for i, tile in enumerate(tiles):
            mask = self.bomb_management_map.get_mask_at_step(self.game_state.tick_number + i)
            # If the next move goes into a dangerous tile, stay still a turn, then continue
            # TODO: we should upgrade the path planning to move in the best non-hit manner (in some cases, moving a
            # different direction could be better)
            if mask[tile] == 0:
                plan_w_bomb_breaks.append('')
            plan_w_bomb_breaks.append(plan[i])

        return plan_w_bomb_breaks, connected

    def greedy_search(self, goal_tile, eval_map, timeout=30): # 20
        tiles = []
        plan = []
        ite = 0
        tile = tuple(goal_tile)
        if not goal_tile or eval_map.value_at_point(tile) == 0:
            return tiles, [], False

        while tile != self.location and ite != timeout:
            moves = np.array([])
            weight = np.array([])
            # Movements
            list_tiles = self.get_cross_tiles(tile)
            (new_tile_r, new_tile_u, new_tile_l, new_tile_d) = (list_tiles[0], list_tiles[1],
                                                                list_tiles[2], list_tiles[3])

            if self.game_state.is_in_bounds(new_tile_r) and eval_map.value_at_point(new_tile_r) > 0:
                moves = np.append(moves, 'r')
                weight = np.append(weight, eval_map.value_at_point(new_tile_r))
            # Up
            if self.game_state.is_in_bounds(new_tile_u) and eval_map.value_at_point(new_tile_u) > 0:
                moves = np.append(moves, 'u')
                weight = np.append(weight, eval_map.value_at_point(new_tile_u))
            # Left
            if self.game_state.is_in_bounds(new_tile_l) and eval_map.value_at_point(new_tile_l) > 0:
                moves = np.append(moves, 'l')
                weight = np.append(weight, eval_map.value_at_point(new_tile_l))
            # Down
            if self.game_state.is_in_bounds(new_tile_d) and eval_map.value_at_point(new_tile_d) > 0:
                moves = np.append(moves, 'd')
                weight = np.append(weight, eval_map.value_at_point(new_tile_d))

            # minimum value
            if weight.size == 0:
                return tiles, [''], False

            index = np.where(weight == np.amin(weight))
            movement = (moves[index].tolist())[-1]
            plan.insert(0, movement)
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
            tiles.insert(0, tile)
        return tiles, plan, True

    @staticmethod
    def is_bomb_connected(b1, b2):
        if abs(b1.position[0] - b2.position[0]) <= 2 or abs(b1.position[1] - b2.position[1]) <= 2:
            return True
        else:
            return False

    def is_ammo_avail(self):
        ammo_tile = self.get_closest_item(self.game_state.ammo)
        if ammo_tile is not None and self.map_representation.value_at_point(ammo_tile) > 0:
            status = True
        else:
            status = False
        return ammo_tile, status

    def is_treasure_avail(self):
        treasure_tile = self.get_closest_item(self.game_state.treasure)
        if treasure_tile is not None and self.map_representation.value_at_point(treasure_tile) > 0:
            status = True
        else:
            status = False
        return treasure_tile, status

    def is_killing_an_option(self):
        tiles_list = []
        for opponent_tile in get_opponents(self.player_state.id, self.game_state._players):
            t = self.get_cross_tiles(opponent_tile)
            tiles_list = tiles_list + t
        bomb_tile = self.evaluate_bomb(tiles_list)
        status = (True if bomb_tile else False)
        return bomb_tile, status

    def is_ore_hot(self):
        tiles_list = []
        for tile in [ore.position for ore in self.ore_counter if ore.counter < 3]:
            t = self.get_cross_tiles(tile)
            tiles_list = tiles_list + t
        bomb_tile = self.evaluate_bomb(tiles_list)
        status = (True if bomb_tile else False)
        return bomb_tile, status

    def get_best_blocking_tile(self): #, tile_list: List[Tuple[int, int]]) -> int:
        safety_map = np.multiply(self.free_map._map, self.map_representation._map)

        safest_tile = np.unravel_index(safety_map.argmax(), safety_map.shape)
        plan, conn = self.plan_to_tile(safest_tile)
        if conn and plan:
            return plan[-1]
        else:
            return ''

    @staticmethod
    def get_cross_tiles(tile: Tuple[int, int]) -> List[Tuple[int, int]]:
        """ From a given tile outputs the neighbour tiles in the 4 directions """
        new_tile_r = (tile[0] - 1, tile[1])
        new_tile_u = (tile[0], tile[1] - 1)
        new_tile_l = (tile[0] + 1, tile[1])
        new_tile_d = (tile[0], tile[1] + 1)
        list_tiles = [new_tile_r, new_tile_u, new_tile_l, new_tile_d]
        return list_tiles

    @staticmethod
    def print_map(game_state):
        """ Print the game map as an array using numpy """
        cols = game_state.size[0]
        rows = game_state.size[1]

        game_map = []
        for x in range(cols):
            for y in range(rows):
                game_map.append((game_state.entity_at((x, y)) if game_state.entity_at((x, y)) is not None else 'f'))

        return game_map
