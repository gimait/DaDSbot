"""
Utility functions
"""

from typing import Optional, Tuple

class OreCounter:
    __slots__ = [
        "counter",
        "position"
    ]
    def __init__(self, position: Tuple[int, int], counter: Optional[int] = 3):
        self.counter = counter
        self.position = position

    def __eq__(self, other):
        if isinstance(other, OreCounter):
            return self.position == other.position
        else:
            return self.position == other

    def got_hit(self):
        self.counter -= 1


def get_opponents(pid, players):
    return [_pos for (_id, _pos) in players if pid != _id]


def manhattan_distance(start, end):

    distance = abs(start[0] - end[0]) + abs(start[1] - end[1])

    return distance

    # given a location as an (x,y) tuple and the bombs on the map
    # we'll return a list of the bomb positions that are nearby


def get_bombs_in_range(location, bombs):

    # empty list to store our bombs that are in range of us
    bombs_in_range = []

    # loop through all the bombs placed in the game
    for bomb in bombs:

        # get manhattan distance to a bomb
        distance = manhattan_distance(location, bomb)

        # set to some arbitrarily high distance
        if distance <= 10:
            bombs_in_range.append(bomb)

    return bombs_in_range

    # given a tile location as an (x,y) tuple, this function
    # will return the surrounding tiles up, down, left and to the right as a list
    # (i.e. [(x1,y1), (x2,y2),...])
    # as long as they do not cross the edge of the map


def get_surrounding_tiles(game_state, location):
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
        if game_state.is_in_bounds(tile):
            # if yes, then add them to our list
            valid_surrounding_tiles.append(tile)

    return valid_surrounding_tiles

    # given a list of tiles
    # return the ones which are actually empty


def get_empty_tiles(game_state, tiles):
    # empty list to store our empty tiles
    empty_tiles = []

    for tile in tiles:
        if not game_state.is_occupied(tile):
            # the tile isn't occupied, so we'll add it to the list
            empty_tiles.append(tile)

    return empty_tiles

    # given a list of tiles and bombs
    # find the tile that's safest to move to


def get_safest_tile(tiles, bombs, location):
    # which bomb is closest to us?
    bomb_distance = 10  # some arbitrary high distance
    closest_bomb = bombs[0]

    for bomb in bombs:
        new_bomb_distance = manhattan_distance(bomb, location)
        if new_bomb_distance < bomb_distance:
            bomb_distance = new_bomb_distance
            closest_bomb = bomb

    safe_dict = {}
    # now we'll figure out which tile is furthest away from that bomb
    for tile in tiles:
        # get the manhattan distance
        distance = manhattan_distance(closest_bomb, tile)
        # store this in a dictionary
        safe_dict[tile] = distance

    # return the tile with the furthest distance from any bomb
    safest_tile = max(safe_dict, key=safe_dict.get)

    return safest_tile

    # given an adjacent tile location, move us there


def move_to_tile(location, tile):
    # actions = ['', 'u', 'd', 'l', 'r', 'p']
    print(f"my tile: {tile}")

    # see where the tile is relative to our current location
    diff = tuple(x - y for x, y in zip(location, tile))

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
