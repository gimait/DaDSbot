"""
Timer classes for bomb explosion and ammo prediction
"""

from typing import Optional, Tuple


class ItemTimer(object):
    """ This class keeps track of the time that a item has been in the map. """
    def __init__(self, step: int, position: Optional[Tuple[int, int]] = None, **kwargs) -> None:
        self.position = position
        self.placement_step = step
        super().__init__(**kwargs)

    def lapsed_time(self, current_step):
        return current_step - self.placement_step

    def __eq__(self, other):
        if isinstance(other, ItemTimer):
            return self.position == other.position
        else:
            return self.position == other


class TimeBomb(ItemTimer):
    """ Timer for bombs (to tell when they will explode)."""
    def time_to_explode(self, current_step):
        return 35 - self.lapsed_time(current_step)


class AmmoTimer(ItemTimer):
    """ Timer for ammo. It can be initialized when a bomb has been created, and updated once the ammo appears."""
    def time_to_appear(self, current_step):
        return 70 - self.placement_step

    def time_to_disappear(self, current_step):
        return 175 - self.placement_step

    def located_at(self, position, step):
        self.position = position
        self.placement_step = step
