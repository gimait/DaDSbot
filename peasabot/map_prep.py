"""
Functions to process the map information
"""

from typing import Tuple, Optional

from coderone.dungeon.agent import GameState

import numpy as np

from .utilities import get_opponents


class GrMap:
    __slots__ = [
        "_map",
        "u_border",
        "v_border",
        "size",
        "player_pos",
        "player_id",
        "__dict__"
    ]

    def __init__(self,
                 map_size: Tuple[int, int],
                 u_border: Optional[np.array] = None,
                 v_border: Optional[np.array] = None) -> None:
        self.size = map_size
        self._map = []
        self.u_border = np.full(self.size[1], -1) if u_border is None else u_border
        self.v_border = np.full((self.size[0] + 2, 1), -1) if v_border is None else v_border

    def _initialize(self):
        self._map = np.zeros(self.state.size)

    def update(self, state: GameState, player_pos: Tuple[int, int], player_id: int) -> None:
        self.state = state
        self.player_pos = player_pos
        self.player_id = player_id
        self._initialize()

    def value_at_point(self, point: Tuple[int, int]) -> int:
        return self._map[point]


class DistanceMap(GrMap):
    """ Creates map of distances from player to all spots"""
    __slots__ = [
        "accessible_area",
    ]

    def __init__(self, map_size: Tuple[int, int]) -> None:
        super().__init__(map_size)
        self.accessible_area = 0

    def _initialize(self) -> np.array:
        # initialize distance map the first time we get one:
        basemap = np.zeros(self.state.size)
        for block in self.state.indestructible_blocks:
            basemap[block] = -1
        for block in self.state.soft_blocks:
            basemap[block] = -1
        for block in self.state.ore_blocks:
            basemap[block] = -1
        for bomb in self.state.bombs:
            basemap[bomb] = -1
        for player in get_opponents(self.player_id, self.state._players):
            basemap[player]
        # Run basic distance with dilation operation
        self._map, self.accessible_area = self._calculate_steps(basemap, self.player_pos)

    def _calculate_steps(self, _map: np.array, center: Tuple[int, int]) -> np.array:
        """ Expand value from center.

            Map expects negative values for blocks and zero for free spots.
        """
        _map[center] = 0.1
        area = 1
        done = False
        # Add borders of map:
        ext_map = np.vstack([self.u_border, _map, self.u_border])
        ext_map = np.hstack([self.v_border, ext_map, self.v_border])

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
                            incr = 1 + min(visible)
                            _map[u - 1, v - 1] = incr
                            ext_map[u, v] = incr
                            area += 1

        for u in range(_map.shape[0]):
            for v in range(_map.shape[1]):
                _map[u, v] = max(0, _map[u, v])
        return _map, area


class BombMap(GrMap):
    __slots__ = [
    ]

    def __init__(self, map_size):
        super().__init__(map_size, v_border=np.full((map_size[0] + 4, 1), -1))

    def _initialize(self) -> np.array:
        basemap = np.zeros(self.state.size)
        for block in self.state.ore_blocks:
            basemap[block] = 1
        for block in self.state.soft_blocks:
            basemap[block] = 1
        for block in self.state.indestructible_blocks:
            basemap[block] = -1
        self._map = self._get_bomb_ranges(basemap)

    def _get_bomb_ranges(self, _map: np.array) -> np.array:
        """Get map with number of blocks affected by bomb in each position.

        Expects a map with 0 for free cells, -1 for undestructible blocks and 1 for destructible blocks.
        """
        # Add borders of map:
        ext_map = np.vstack([self.u_border, self.u_border, _map, self.u_border, self.u_border])
        ext_map = np.hstack([self.v_border, self.v_border, ext_map, self.v_border, self.v_border])

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


class FreedomMap(GrMap):
    __slots__ = [
        "mask"
    ]

    def __init__(self, map_size):
        super().__init__(map_size, u_border=np.full(map_size[1], 0), v_border=np.full((map_size[0] + 2, 1), 0))
        self.mask = np.array(((1, 2, 1), (2, 0, 2), (1, 2, 1)))

    def _initialize(self) -> np.array:
        basemap = np.ones(self.state.size)
        for block in self.state.indestructible_blocks:
            basemap[block] = 0
        for block in self.state.soft_blocks:
            basemap[block] = 0
        for block in self.state.ore_blocks:
            basemap[block] = 0
        for bomb in self.state.bombs:
            basemap[bomb] = 0
        for player in get_opponents(self.player_id, self.state._players):
            basemap[player] = 0
        self._map = self._grad_convolution(basemap)

    def _grad_convolution(self, _map: np.array) -> np.array:

        ext_map = np.vstack([self.u_border, _map, self.u_border])
        ext_map = np.hstack([self.v_border, ext_map, self.v_border])

        for u in range(1, ext_map.shape[0] - 1):
            for v in range(1, ext_map.shape[1] - 1):
                if ext_map[u, v] == 1:
                    val = 0
                    for i in range(3):
                        for j in range(3):
                            val += self.mask[i, j] * ext_map[u - 1 + i, v - 1 + j]
                    _map[u - 1, v - 1] = val
        return _map


def gen_manhattan_map(base_size: Tuple[int, int]) -> np.array:
    """Generate a supermap that contains all possible minimum distances to points in the map"""
    mega_map = np.zeros((base_size[0] * 2 - 1, base_size[1] * 2 - 1))
    for u in range(mega_map.shape[0]):
        for v in range(mega_map.shape[1]):
            mega_map[u, v] = abs(base_size[0] - u) + abs(base_size[1] - v)
    return mega_map
