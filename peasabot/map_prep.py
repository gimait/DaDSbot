"""
Functions to process the map information
"""

from typing import Tuple

from coderone.dungeon.agent import GameState, PlayerState

import numpy as np


class DistanceMap:
    """ Creates map of distances from player to all spots"""
    __slots__ = [
        "distance_map",
        "accessible_area",
        "size",
        "player_pos",
        "player_id",
        "__dict__"
    ]

    def __init__(self, map_size: Tuple[int, int]) -> None:
        self.size = map_size
        self.accessible_area = 0
        self.distance_map = []

    def update(self, state: GameState, player_pos: Tuple[int, int], player_id: int) -> None:
        self.state = state
        self.player_pos = player_pos
        self.player_id = player_id
        self.distance_map, self.accessible_area = self._init_distance_map()

    def distance_to_point(self, point: Tuple[int, int]) -> int:
        return self.distance_map[point]

    def _init_distance_map(self) -> np.array:
        # initialize distance map the first time we get one:
        basemap = np.zeros(self.state.size)
        for block in self.state.all_blocks:
            basemap[block] = -1
        for bomb in self.state.bombs:
            basemap[bomb] = -1
        for player in self.state.opponents(self.player_id):
            basemap[player]
        # Run basic distance with dilation operation
        return self._calculate_steps(basemap, self.player_pos)

    @staticmethod
    def _calculate_steps(_map: np.array, center: Tuple[int, int]) -> np.array:
        """ Expand value from center.

            Map expects negative values for blocks and zero for free spots.
        """
        _map[center] = 0.1
        area = 1
        done = False
        # Add borders of map:
        u_ones = np.full(_map.shape[1], -1)
        v_ones = np.full((_map.shape[0] + 2, 1), -1)
        ext_map = np.vstack([u_ones, _map, u_ones])
        ext_map = np.hstack([v_ones, ext_map, v_ones])

        while not done:
            done = True
            for u in range(1, ext_map.shape[0] - 1):
                for v in range(1, ext_map.shape[1] - 1):
                    if ext_map[u, v] == 0:
                        visible = []
                        if ext_map[u - 1, v] > 0:
                            visible.append(ext_map[u - 1, v])
                        if ext_map[u + 1, v] > 0:
                            visible.append(ext_map[u + 1, v])
                        if ext_map[u, v - 1] > 0:
                            visible.append(ext_map[u, v - 1])
                        if ext_map[u, v + 1] > 0:
                            visible.append(ext_map[u, v + 1])
                        if visible:
                            done = False
                            incr = 1 + max(visible)
                            _map[u - 1, v - 1] = incr
                            ext_map[u, v] = incr
                            area += 1

        for u in range(_map.shape[0]):
            for v in range(_map.shape[1]):
                _map[u, v] = max(0, _map[u, v])
        return _map, area


class BombMap:
    __slots__ = [
        "bomb_effect_map",
        "__dict__"
    ]

    def __init__(self):
        self.bomb_effect_map = []

    def update(self, state: GameState, player: PlayerState) -> None:
        self.state = state
        self.player = player
        self.bomb_effect_map = self._init_bomb_effect_map()

    def _init_bomb_effect_map(self) -> np.array:
        basemap = np.zeros(self.state.size)
        for block in self.state.ore_blocks:
            basemap[block] = 1
        for block in self.state.soft_blocks:
            basemap[block] = 1
        for block in self.state.indestructible_blocks:
            basemap[block] = -1
        return self._get_bomb_ranges(basemap)

    @staticmethod
    def _get_bomb_ranges(_map: np.array) -> np.array:
        """Get map with number of blocks affected by bomb in each position.

        Expects a map with 0 for free cells, -1 for undestructible blocks and 1 for destructible blocks.
        """
        # Add borders of map:
        u_ones = np.full(_map.shape[1], -1)
        v_ones = np.full((_map.shape[0] + 4, 1), -1)
        ext_map = np.vstack([u_ones, u_ones, _map, u_ones, u_ones])
        ext_map = np.hstack([v_ones, v_ones, ext_map, v_ones, v_ones])

        for u in range(2, ext_map.shape[0] - 2):
            for v in range(2, ext_map.shape[1] - 2):
                if ext_map[u, v] == 0:
                    targets = 0
                    if ext_map[u - 1, v] > 0:
                        targets += 1
                    elif ext_map[u - 2, v] > 0:
                        targets += 1
                    if ext_map[u + 1, v] > 0:
                        targets += 1
                    elif ext_map[u + 2, v] > 0:
                        targets += 1
                    if ext_map[u, v - 1] > 0:
                        targets += 1
                    elif ext_map[u, v - 2] > 0:
                        targets += 1
                    if ext_map[u, v + 1] > 0:
                        targets += 1
                    elif ext_map[u, v + 1] > 0:
                        targets += 1
                    if targets:
                        _map[u - 2, v - 2] = targets
        return _map


def gen_manhattan_map(base_size: Tuple[int, int]) -> np.array:
    """Generate a supermap that contains all possible minimum distances to points in the map"""
    mega_map = np.zeros((base_size[0] * 2 - 1, base_size[1] * 2 - 1))
    for u in range(mega_map.shape[0]):
        for v in range(mega_map.shape[1]):
            mega_map[u, v] = abs(base_size[0] - u) + abs(base_size[1] - v)
    return mega_map
