"""
Functions to process the map information
"""

from typing import Tuple, Optional

from coderone.dungeon.agent import GameState

import numpy as np

from .timers import TimeBomb
from .utilities import get_opponents


class GrMap(object):
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
                 size: Tuple[int, int],
                 u_border: Optional[np.array] = None,
                 v_border: Optional[np.array] = None, **kwargs) -> None:
        self.size = size
        self._map = []
        self.u_border = np.full(self.size[1], -1) if u_border is None else u_border
        self.v_border = np.full((self.size[0] + 2, 1), -1) if v_border is None else v_border
        super().__init__(**kwargs)

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

    def __init__(self, size: Tuple[int, int]) -> None:
        super().__init__(size)
        self.accessible_area = 0
        self.distance_penalty_map = []
        self.accessible_area_mask = []

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
            basemap[player] = -1
        # Run basic distance with dilation operation
        (self.accessible_area, self._map,
         self.distance_penalty_map, self.accessible_area_mask) = self._calculate_steps(basemap, self.player_pos)

    def _calculate_steps(self, base_map: np.array, center: Tuple[int, int]) -> Tuple[int, np.array, np.array, np.array]:
        """ Expand value from center.

            Map expects negative values for blocks and zero for free spots.
        """
        inv_map = np.zeros(base_map.shape)
        mask_map = np.zeros(base_map.shape)
        base_map[center] = 0.1
        inv_map[center] = 0.1
        mask_map[center] = 1

        area = 1
        done = False
        # Add borders of map:
        ext_map = np.vstack([self.u_border, base_map, self.u_border])
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
                            base_map[u - 1, v - 1] = incr
                            inv_map[u - 1, v - 1] = 1 / incr
                            mask_map[u - 1, v - 1] = 1
                            ext_map[u, v] = incr
                            area += 1

        for u in range(base_map.shape[0]):
            for v in range(base_map.shape[1]):
                base_map[u, v] = max(0, base_map[u, v])
        return area, base_map, inv_map, mask_map


class TargetMap(GrMap):
    __slots__ = [
    ]

    def __init__(self, size):
        super().__init__(size, v_border=np.full((size[0] + 4, 1), -1))

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
                    elif ext_map[u - 1, v] == 0 and ext_map[u - 2, v] > 0:
                        targets += 1
                    if ext_map[u + 1, v] > 0:
                        targets += 1
                    elif ext_map[u + 1, v] == 0 and ext_map[u + 2, v] > 0:
                        targets += 1
                    if ext_map[u, v - 1] > 0:
                        targets += 1
                    elif ext_map[u, v - 1] == 0 and ext_map[u, v - 2] > 0:
                        targets += 1
                    if ext_map[u, v + 1] > 0:
                        targets += 1
                    elif ext_map[u, v + 1] == 0 and ext_map[u, v + 2] > 0:
                        targets += 1
                    if targets:
                        _map[u - 2, v - 2] = targets * targets
        return _map


class FreedomMap(GrMap):
    __slots__ = [
        "mask"
    ]

    def __init__(self, size):
        super().__init__(size, u_border=np.full(size[1], 0), v_border=np.full((size[0] + 2, 1), 0))
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
                    _map[u - 1, v - 1] = val * val
        return _map


class BombAreaMap(GrMap, TimeBomb):
    def __init__(self, size, step: int, position: Tuple[int, int]):
        super().__init__(size=size, v_border=np.full((size[0] + 4, 1), -1), step=step, position=position)
        self._idx_cross = [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (0, 2), (0, 1), (0, -1), (0, -2)]
        self._initialize()

    def _initialize(self):
        self._map = np.zeros(self.size)
        self._add_cross_to_map(self.position)

    def update(self, new_bomb: Tuple[int, int]) -> bool:
        if self._map[new_bomb]:
            self._add_cross_to_map(new_bomb)
            return True
        else:
            return False

    def _add_cross_to_map(self, tile) -> None:
        # Add borders of map:
        for c in self._idx_cross:
            affected_tile = (tile[0] + c[0], tile[1] + c[1])
            if self.size[0] > affected_tile[0] >= 0 and self.size[1] > affected_tile[1] >= 0:
                self._map[affected_tile] = -1


def gen_manhattan_map(base_size: Tuple[int, int]) -> np.array:
    """Generate a supermap that contains all possible minimum distances to points in the map"""
    mega_map = np.zeros((base_size[0] * 2 - 1, base_size[1] * 2 - 1))
    for u in range(mega_map.shape[0]):
        for v in range(mega_map.shape[1]):
            mega_map[u, v] = abs(base_size[0] - u) + abs(base_size[1] - v)
    return mega_map
