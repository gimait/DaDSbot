"""
Functions to process the map information
"""

from typing import Optional, Tuple

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

    def _initialize(self, mask: Optional[np.array] = None):
        self._map = np.zeros(self.state.size) + mask

    def update(self, state: GameState, player_pos: Tuple[int, int], player_id: int) -> None:
        self.state = state
        self.player_pos = player_pos
        self.player_id = player_id

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

    def update(self,
               state: GameState, player_pos: Tuple[int, int], player_id: int,
               mask: Optional[np.array] = None) -> None:
        super().update(state, player_pos, player_id)
        self._initialize(mask)

    def _initialize(self, mask: Optional[np.array] = None) -> None:
        # initialize distance map the first time we get one:
        basemap = np.zeros(self.state.size)
        for block in self.state.indestructible_blocks:
            basemap[block] = -1
        for block in self.state.soft_blocks:
            basemap[block] = -1
        for block in self.state.ore_blocks:
            basemap[block] = -1
        if mask is not None:
            basemap += mask - 1
        for bomb in self.state.bombs:
            basemap[bomb] = -1
        for player in get_opponents(self.player_id, self.state._players):
            basemap[player] = -1
        # Run basic distance with dilation operation
        (self.accessible_area, self._map,
         self.distance_penalty_map, self.accessible_area_mask) = self._calculate_steps(basemap, self.player_pos)

        ore_free_map = basemap.copy()
        for block in self.state.ore_blocks:
            ore_free_map[block] = 0
        (_, _, self.ore_penalty_map, _) = self._calculate_steps(ore_free_map, self.player_pos)

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

    def update(self,
               state: GameState, player_pos: Tuple[int, int], player_id: int,
               mask: Optional[np.array] = None) -> None:
        super().update(state, player_pos, player_id)
        self._initialize(mask)

    def _initialize(self, mask: Optional[np.array] = None) -> None:
        basemap = np.zeros(self.state.size)
        for block in self.state.ore_blocks:
            basemap[block] = 1
        for block in self.state.soft_blocks:
            basemap[block] = 1
        for block in self.state.indestructible_blocks:
            basemap[block] = -1
        if mask is not None:
            basemap += mask - 1
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
        self.mask0 = np.array(((0, 2, 0),
                               (2, 0, 2),
                               (0, 2, 0)))

        self.mask1 = np.array(((1, 0, 1),
                               (0, 0, 0),
                               (1, 0, 1)))

    def update(self,
               state: GameState, player_pos: Tuple[int, int], player_id: int,
               mask: Optional[np.array] = None) -> None:
        super().update(state, player_pos, player_id)
        self._initialize(mask)

    def _initialize(self, mask: Optional[np.array] = None) -> None:
        basemap = np.ones(self.state.size)
        for block in self.state.indestructible_blocks:
            basemap[block] = 0
        for block in self.state.soft_blocks:
            basemap[block] = 0
        for block in self.state.ore_blocks:
            basemap[block] = 0
        if mask is not None:
            basemap += mask - 1
        else:
            for bomb in self.state.bombs:
                basemap[bomb] = -1
        # for player in get_opponents(self.player_id, self.state._players):
        #     basemap[player] = 0
        self._map = self._grad_convolution(basemap)

    def _grad_convolution(self, _map: np.array) -> np.array:

        ext_map = np.vstack([self.u_border, _map, self.u_border])
        ext_map = np.hstack([self.v_border, ext_map, self.v_border])

        for u in range(1, ext_map.shape[0] - 1):
            for v in range(1, ext_map.shape[1] - 1):
                if ext_map[u, v] == 1:
                    val = self._apply_mask(ext_map, (u, v), self.mask0)
                    if val > 4:
                        val += self._apply_mask(ext_map, (u, v), self.mask1)
                    _map[u - 1, v - 1] = val ** 2
        return _map

    @staticmethod
    def _apply_mask(_map, center, mask):
        val = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                val += mask[i, j] * _map[center[0] - 1 + i, center[1] - 1 + j]
        return val


class BombArea(GrMap, TimeBomb):
    """
    Takes care of managing the bombs with their timers.
    Contains a map of zeros with ones on the areas of effect of a single bomb.
    The update method allows to change the time of placement of the bomb, which is related to the time of explosion.
    """

    __slots__ = [
        "_idx_cross",
        "affected_area",
        "fired",
        "owned"
    ]

    def __init__(self, size, step: int, position: Tuple[int, int], owned: bool, danger_thresh: int):
        super().__init__(size=size, v_border=np.full((size[0] + 4, 1), -1), step=step, position=position)
        self._idx_cross = [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (0, 2), (0, 1), (0, -1), (0, -2)]
        self.affected_area = 0
        self.fired = False
        self.owned = owned
        self.counted = False  # This is to check if we already accounted for the ore being hit by this one
        self.danger_thresh = danger_thresh
        self._initialize()

    def _initialize(self):
        self._map = np.zeros(self.size)
        self._add_cross_to_map(self.position)

    def update(self, new_time: int, owned: bool) -> None:
        """ Update the time of explosion of the bomb."""
        # If the new time is smaller than the old time, the ownership changes
        if new_time < self.placement_step:
            self.placement_step = new_time
            self.owned = owned

    def should_be_avoided(self, tick):
        if self.time_to_explode(tick) < self.danger_thresh: #(self.affected_area / 4):
            return True
        else:
            return False

    def _add_cross_to_map(self, tile) -> None:
        # Add borders of map:
        for c in self._idx_cross:
            affected_tile = (tile[0] + c[0], tile[1] + c[1])
            if self.size[0] > affected_tile[0] >= 0 and self.size[1] > affected_tile[1] >= 0 \
               and self._map[affected_tile] >= 0:
                self._map[affected_tile] = 1
                self.affected_area += 1


class BombAreaMap(GrMap):
    """
    This class encapsulates and manages all bombs, making available a set of masks indicating the areas of your own
    bombs, other persons bombs and the danger areas given a threshold, as a map of ones with the affected area as zeros.
    """
    __slots__ = [
        "bombs",
        "danger_thresh"
    ]

    def __init__(self, size: Tuple[int, int], danger_thresh: Optional[int] = 1):
        super().__init__(size)
        self.bombs = []
        self.danger_thresh = danger_thresh
        self.in_danger = False
        self.last_placed_bomb = None

    def update(self, game_state: GameState, player_pos: Tuple[int, int]) -> None:

        # First, add new bombs to list
        for game_bomb in game_state.bombs:
            if game_bomb not in self.bombs:
                own = player_pos == game_bomb
                ba = BombArea(size=self.size,
                              step=game_state.tick_number,
                              position=game_bomb,
                              owned=own,
                              danger_thresh=self.danger_thresh)
                if own:
                    self.last_placed_bomb = ba
                else:
                    self.last_placed_bomb = None
                self.bombs.append(ba)

        # Then, update the bombs
        i = 0
        while i < len(self.bombs):
            # If the bomb was fired, get rid of it
            if self.bombs[i].fired:
                del self.bombs[i]
            else:
                # Otherwise, we need to update the timing of all connected bombs
                if len(self.bombs) > 1:
                    for other_bomb in self.bombs[1:]:
                        if self.bombs[i]._map[other_bomb.position] > 0:
                            other_bomb.update(new_time=(self.bombs[i].placement_step + 1), owned=self.bombs[i].owned)
                # On top of that, if this bomb is no longer in the list given by the game, it means that it was fired
                if self.bombs[i] not in game_state.bombs:
                    self.bombs[i].fired = True
                i += 1

        # Once we updated the info about all bombs, let's generate a danger mask, another with our bombs areas and
        # another with
        self._update_maps(game_state.tick_number)

    def _update_maps(self, tick_number):
        # TODO: The owned/ not owned maps don't differenciate when they collide on who would take the bomb.
        # It would be possible that placing a bomb in one of those conflictive areas we lost all of our area or
        # we stole it. We should add some checks for this, to ensure that the areas are well defined.
        danger_map = np.zeros(self.size)
        owned_map = np.zeros(self.size)
        not_owned_map = np.zeros(self.size)
        all_map = np.zeros(self.size)
        self.in_danger = False
        for bomb in self.bombs:
            all_map += bomb._map
            if bomb.owned:
                owned_map += bomb._map
            else:
                not_owned_map += bomb._map
            if bomb.should_be_avoided(tick_number):
                self.in_danger = True
                danger_map += bomb._map

        self._map = np.where(owned_map > 0, 0, 1)
        self.opponent = np.where(not_owned_map > 0, 0, 1)
        self.danger_zone = np.where(danger_map > 0, 0, 1)
        self.all_map = np.where(all_map > 0, 0, 1)

    def is_in_danger_at(self, tile: Tuple[int,int]):
        col = tile[0]
        row = tile[1]
        if self.danger_zone[col][row] == 1:
            return self.danger_zone, False
        else:
            return self.danger_zone, True

    def get_mask_at_step(self, step: int) -> np.array:
        mask = np.zeros(self.size)
        for bomb in self.bombs:
            if bomb.time_to_explode(step) == 0:
                mask += bomb._map
        return np.where(mask > 0, 0, 1)

    def set_danger_threshold(self, threshold):
        self.danger_thresh = threshold
