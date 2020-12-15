"""
Consumer bot
"""

import random
from typing import List, Optional, Tuple

from coderone.dungeon.agent import GameState, PlayerState

import numpy as np

from .map_prep import BombAreaMap, DistanceMap, FreedomMap, TargetMap
from .utilities import get_opponents

TIMEOUT = 20  # Maximum of positions to calculate in the planning
MAX_BOMB = 5  # Don't pick up more
MIN_BOMB = 1  # Don't place bomb
BOMB_TICK_THRESHOLD = 0  # Time for escaping of the bomb
CORNER_THRESH = 30  # Threshold that indicates when a spot has a very low degree of freedom
ATTACK_THRESH = 70
DANGER_THRESH = 0


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
        self.bomb_target_map = TargetMap(self.size)
        self.bomb_management_map = BombAreaMap(self.size, danger_thresh=BOMB_TICK_THRESHOLD)
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

        self.opponent_tile = get_opponents(self.player_state.id, self.game_state._players)[0]

        # Check new bombs and update timers
        self.bomb_management_map.update(game_state, player_state.location)
        self.free_map.update(game_state, player_state.location, player_state.id)
        self.bomb_target_map.update(game_state, player_state.location, player_state.id)
        self.map_representation.update(game_state, player_state.location, player_state.id)

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
        optimal_points = np.multiply(self.map_representation.distance_penalty_map, self.bomb_target_map._map)
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

    def path_to_safest_area(self, danger_zone: Optional[BombAreaMap] = None):
        # take out unsafe tiles from free_map:
        if danger_zone is not None:
            safety_map = np.multiply(danger_zone,
                                     np.multiply(self.free_map._map,
                                                 self.map_representation.distance_penalty_map))
        else:
            safety_map = np.multiply(self.free_map._map, self.map_representation._map)
        safest_tile = np.unravel_index(safety_map.argmax(), safety_map.shape)

        return self.plan_to_tile(safest_tile)

    def path_to_freest_area(self, danger_zone: Optional[BombAreaMap] = None):
        # take out unsafe tiles from free_map:
        if danger_zone is not None:
            safety_map = np.multiply(danger_zone,
                                     np.multiply(self.free_map._map,
                                                 self.map_representation.accessible_area_mask))
        else:
            safety_map = np.multiply(self.free_map._map, self.map_representation._map)
        safest_tile = np.unravel_index(safety_map.argmax(), safety_map.shape)

        return self.plan_to_tile(safest_tile)

    def plan_to_tile(self, goal_tile: Tuple[int, int]) -> Tuple[List, bool]:
        if not goal_tile:
            return [], False

        plan = []
        tile = tuple(goal_tile)
        if self.map_representation.value_at_point(tile) == 0:
            return plan, False

        tiles = []
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

        plan_w_bomb_breaks = []
        for i, tile in enumerate(tiles):
            mask = self.bomb_management_map.get_mask_at_step(self.game_state.tick_number + i)
            # If the next move goes into a dangerous tile, stay still a turn, then continue
            # TODO: we should upgrade the path planning to move in the best non-hit manner (in some cases, moving a
            # different direction could be better)
            if mask[tile] == 0:
                plan_w_bomb_breaks.append('')
            plan_w_bomb_breaks.append(plan[i])

        return plan_w_bomb_breaks, True

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

    def get_best_blocking_tile(self, tile_list: List[Tuple[int, int]]) -> int:
        safety_map = np.multiply(self.free_map._map, self.map_representation._map)

        safest_tile = np.unravel_index(safety_map.argmax(), safety_map.shape)
        plan, conn = self.plan_to_tile(safest_tile)
        if conn and plan:
            return plan[-1]
        else:
            return ''

    def next_move_killer(self):
        # Agent possibilities
        danger_zone, danger_status = self.bomb_management_map.is_in_danger()
        ammo_tile, ammo_status = self.is_ammo_avail()
        treasure_tile, treasure_status = self.is_treasure_avail()
        kill_tiles, kill_status = self.is_killing_an_option()

        # 1 Avoid Danger
        if danger_status:
            plan, _ = self.path_to_safest_area(danger_zone)
        # 2 Pick up ammo if less than MAX
        elif ammo_status and self.ammo < MAX_BOMB:
            plan, _ = self.plan_to_tile(ammo_tile)
        # 3 Pick up treasures, they are also good
        elif treasure_status:
            plan, _ = self.plan_to_tile(treasure_tile)
        # 4 If we finished a plan, get away
        elif self.previous_plan == "run":
            d = danger_zone if self.bomb_management_map.last_placed_bomb is None \
                else danger_zone - self.bomb_management_map.last_placed_bomb._map
            plan, _ = self.path_to_safest_area(d)
            self.previous_plan = None
        # 4 Plan for killing, finish it if started
        elif (0 < self.free_map._map[self.opponent_tile] < ATTACK_THRESH) and \
             (self.previous_plan == "kill" or kill_status):
            plan, connected = self.plan_to_tile(kill_tiles)
            self.previous_plan = (None if not plan else "kill")
            plan.append('p')
        # 5 Place a bomb in a good place if you have bombs
        elif self.previous_plan == "loot" or self.ammo > MIN_BOMB:
            best_point_for_bomb = self.get_best_point_for_bomb()
            plan, _ = self.plan_to_tile(best_point_for_bomb)
            self.previous_plan = (None if not plan else "loot")
            plan.append('p')
        # 6 If there is still ammo around and we are bored, let's go catch it
        elif ammo_status:
            plan, _ = self.plan_to_tile(ammo_tile)

        # Last Find a good place to wait
        else:
            free_tile = self.get_freedom_tiles()
            plan, _ = self.plan_to_tile(free_tile)

        if plan and not self.is_step_safe(plan[-1]):
            return ['']
        return plan

    def is_step_safe(self, step):
        if not self.current_bombs:
            return True

        future_pos = ()
        if step == 'd':
            future_pos = (self.location[0], self.location[1] - 1)
        elif step == 'l':
            future_pos = (self.location[0] - 1, self.location[1])
        elif step == 'u':
            future_pos = (self.location[0], self.location[1] + 1)
        elif step == 'r':
            future_pos = (self.location[0] + 1, self.location[1])
        else:
            return True

        if not self.game_state.is_in_bounds(future_pos) and self.map_representation.accessible_area[future_pos]:
            return False

        tte = self.current_bombs[0].time_to_explode(self.game_state.tick_number)
        if tte <= 2 and self.current_bombs[0]._map[future_pos] < 0:
            return False
        return True

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
