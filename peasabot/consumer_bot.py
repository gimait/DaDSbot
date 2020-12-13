"""
Consumer bot
"""

import random
from typing import List, Optional, Tuple

from coderone.dungeon.agent import GameState, PlayerState

import numpy as np

from .timers import TimeBomb
from .utilities import get_opponents

TIMEOUT = 20  # Maximum of positions to calculate in the planning


class ConsumerBot():

    def update_state(self, game_state: GameState, player_state: PlayerState):
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

        # get list of bombs within range
        self.bombs_in_range = self.get_bombs_in_range(self.location, game_state.bombs)

        # get our surrounding tiles
        self.surrounding_tiles = self.get_surrounding_tiles(self.location)

        # get list of empty tiles around us
        self.empty_tiles = self.get_empty_tiles(self.surrounding_tiles)

        ########################
        # ##    HELPERS     ## #
        ########################

    def manhattan_distance(self, start, end):

        distance = abs(start[0] - end[0]) + abs(start[1] - end[1])

        return distance

        # given a location as an (x,y) tuple and the bombs on the map
        # we'll return a list of the bomb positions that are nearby

    def get_bombs_in_range(self, location, bombs):

        # empty list to store our bombs that are in range of us
        bombs_in_range = []

        # loop through all the bombs placed in the game
        for bomb in bombs:

            # get manhattan distance to a bomb
            distance = self.manhattan_distance(location, bomb)

            # set to some arbitrarily high distance
            if distance <= 10:
                bombs_in_range.append(bomb)

        return bombs_in_range

        # given a tile location as an (x,y) tuple, this function
        # will return the surrounding tiles up, down, left and to the right as a list
        # (i.e. [(x1,y1), (x2,y2),...])
        # as long as they do not cross the edge of the map

    def get_surrounding_tiles(self, location):
        # find all the surrounding tiles relative to us
        # location[0] = col index; location[1] = row index
        tile_up = (location[0], location[1] + 1)
        tile_down = (location[0], location[1] - 1)
        tile_left = (location[0] - 1, location[1])
        tile_right = (location[0] + 1, location[1])

        # combine these into a list
        all_surrounding_tiles = [tile_up, tile_down, tile_left, tile_right]

        # we'll need to remove tiles that cross the border of the map
        # start with an empty list to store our valid surrounding tiles
        valid_surrounding_tiles = []

        # loop through our tiles
        for tile in all_surrounding_tiles:
            # check if the tile is within the boundaries of the game
            if self.game_state.is_in_bounds(tile):
                # if yes, then add them to our list
                valid_surrounding_tiles.append(tile)

        return valid_surrounding_tiles

        # given a list of tiles
        # return the ones which are actually empty

    def get_empty_tiles(self, tiles):
        # empty list to store our empty tiles
        empty_tiles = []

        for tile in tiles:
            if not self.game_state.is_occupied(tile):
                # the tile isn't occupied, so we'll add it to the list
                empty_tiles.append(tile)

        return empty_tiles

        # given a list of tiles and bombs
        # find the tile that's safest to move to

    def get_safest_tile(self, tiles, bombs):
        # which bomb is closest to us?
        bomb_distance = 10  # some arbitrary high distance
        closest_bomb = bombs[0]

        for bomb in bombs:
            new_bomb_distance = self.manhattan_distance(bomb, self.location)
            if new_bomb_distance < bomb_distance:
                bomb_distance = new_bomb_distance
                closest_bomb = bomb

        safe_dict = {}
        # now we'll figure out which tile is furthest away from that bomb
        for tile in tiles:
            # get the manhattan distance
            distance = self.manhattan_distance(closest_bomb, tile)
            # store this in a dictionary
            safe_dict[tile] = distance

        # return the tile with the furthest distance from any bomb
        safest_tile = max(safe_dict, key=safe_dict.get)

        return safest_tile

        # given an adjacent tile location, move us there

    def move_to_tile(self, location, tile):
        # actions = ['', 'u', 'd', 'l', 'r', 'p']
        print(f"my tile: {tile}")

        # see where the tile is relative to our current location
        diff = tuple(x - y for x, y in zip(self.location, tile))

        # return the action that moves in the direction of the tile
        if diff == (0, 1):
            action = 'd'
        elif diff == (1, 0):
            action = 'l'
        elif diff == (0, -1):
            action = 'u'
        elif diff == (-1, 0):
            action = 'r'
        else:
            action = ''

        return action

    @staticmethod
    def print_map(game_state):
        """ Print the game map as an array using numpy """
        cols = game_state.size[0]
        rows = game_state.size[1]

        game_map = np.zeros((rows, cols))

        for x in cols:
            for y in rows:
                entity = game_state.entity_at((x, y))
                if entity is not None:
                    game_map[x][y] = entity
                else:
                    game_map[x][y] = 'f'  # free space

        return game_map

    def get_cross_tiles(self, tile):
        """ From a given tile outputs the neighbour tiles in the 4 directions """
        new_tile_r = (tile[0] - 1, tile[1])
        new_tile_u = (tile[0], tile[1] - 1)
        new_tile_l = (tile[0] + 1, tile[1])
        new_tile_d = (tile[0], tile[1] + 1)
        list_tiles = [new_tile_r, new_tile_u, new_tile_l, new_tile_d]
        return list_tiles

    def get_closest_ammo(self):
        distance = 999  # here check with the value of the no possible from the distance_to_point
        tile = None
        for i, ammo_tile in enumerate(self.game_state.ammo):
            d2p_ammo = self.map_representation.distance_to_point(ammo_tile)
            if d2p_ammo != 0 and d2p_ammo < distance:
                distance = d2p_ammo
                tile = ammo_tile
        return tile

    def evaluate_bomb(self, tiles_list):
        # Input a list of tiles to evaluate Outputs the best tile to place a bomb
        # First it only considers the closest distance if any
        weight = np.array([])
        tlist = ([t for t in tiles_list if self.game_state.is_in_bounds(t)])
        tnplist = np.array(tlist)
        if not tlist:
            return([])
        for tiles in tlist:
            self.map_representation.distance_to_point(tiles)
            weight = np.append(weight, self.map_representation.distance_to_point(tiles))
        index = np.where(weight == np.amin(weight))
        tile = (tnplist[index].tolist())[-1]
        return tile

    def plan_to_tile(self, goal_tile):
        plan = []
        tile = goal_tile

        timeout = TIMEOUT
        ite = 0
        while tile != self.location and ite != timeout:
            moves = np.array([])
            weight = np.array([])
            # Movements
            list_tiles = self.get_cross_tiles(tile)
            (new_tile_r, new_tile_u, new_tile_l, new_tile_d) = (list_tiles[0], list_tiles[1],
                                                                list_tiles[2], list_tiles[3])
            if self.game_state.is_in_bounds(new_tile_r) and self.map_representation.distance_to_point(new_tile_r) > 0:
                moves = np.append(moves, 'r')
                weight = np.append(weight, self.map_representation.distance_to_point(new_tile_r))
            # Up
            if self.game_state.is_in_bounds(new_tile_u) and self.map_representation.distance_to_point(new_tile_u) > 0:
                moves = np.append(moves, 'u')
                weight = np.append(weight, self.map_representation.distance_to_point(new_tile_u))
            # Left
            if self.game_state.is_in_bounds(new_tile_l) and self.map_representation.distance_to_point(new_tile_l) > 0:
                moves = np.append(moves, 'l')
                weight = np.append(weight, self.map_representation.distance_to_point(new_tile_l))
            # Down
            if self.game_state.is_in_bounds(new_tile_d) and self.map_representation.distance_to_point(new_tile_d) > 0:
                moves = np.append(moves, 'd')
                weight = np.append(weight, self.map_representation.distance_to_point(new_tile_d))

            # minimum value
            if weight.size == 0:
                return([''])

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
        return plan

    def next_move_killer(self):
        # First, check that you are not in a danger zone
        bombs_about_to_explode = []
        for b in self.current_bombs:
            if b.time_to_explode(self.game_state.tick_number) < 10:
                bombs_about_to_explode.append(b.position)

        for b in bombs_about_to_explode:
            if abs(b[0] - self.location[0]) <= 2 or abs(b[1] - self.location[1]):
                danger_zone = [b, (b[0] - 2, b[1]), (b[0] - 1, b[1]), (b[0] + 2, b[1]), (b[0] + 1, b[1]),
                               (b[0], b[1] - 2), (b[0], b[1] - 1), (b[0], b[1] + 2), (b[0], b[1] + 1)]
                plan = self.path_to_safest_area(danger_zone)
                return plan

        if self.substrategy == 1:
            # Pick up bombS
            ammo_tile = self.get_closest_ammo()
            if ammo_tile is not None and self.map_representation.distance_to_point(ammo_tile) > 0:
                plan = self.plan_to_tile(ammo_tile)
                return(plan)
            else:
                self.substrategy = 2

        if self.substrategy == 2:
            # Harassment and place
            if self.player_state.ammo <= 2:
                self.substrategy = 1
            # Go towards the closer player
            tiles_list = []
            for opponent_tile in get_opponents(self.player_state.id, self.game_state._players):
                t = self.get_cross_tiles(opponent_tile)
                tiles_list = tiles_list + t
            if self.player_state.location in tiles_list:
                self.substrategy = 1
                return(['p'])
            bomb_tile = self.evaluate_bomb(tiles_list)
            if bomb_tile:
                plan = self.plan_to_tile(bomb_tile)
                plan.insert(0, 'p')
                return (plan)

        return ['']  # [random.choice(self.actions)]

    def path_to_safest_area(self, danger_zone: Optional[List[Tuple[int, int]]]):
        # take out unsafe tiles from free_map:
        for tile in danger_zone:
            if tile[0] >= 0 and tile[1] >= 0 and tile[0] < self.free_map.size[0] and tile[1] < self.free_map.size[1]:
                self.free_map._map[tile] = 0
        safety_map = np.multiply(self.free_map._map, self.map_representation._map)

        safest_tile = np.unravel_index(safety_map.argmax(), safety_map.shape)

        return self.plan_to_tile(safest_tile)

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
