"""
Timer classes for bomb explosion and ammo prediction
"""

from typing import Optional, Tuple


class ItemTimer:
    __slots__ = [
        "position",
        "placement_step"
    ]

    def __init__(self, step: int, position: Optional[Tuple[int, int]] = None) -> None:
        self.position = position
        self.placement_step = step

    def lapsed_time(self, current_step):
        return current_step - self.placement_step


class TimeBomb(ItemTimer):
    def time_to_explode(self, current_step):
        return 35 - self.lapsed_time(current_step)


class AmmoTimer(ItemTimer):
    def time_to_appear(self, current_step):
        return 70 - self.placement_step

    def time_to_disappear(self, current_step):
        return 175 - self.placement_step

    def located_at(self, position, step):
        self.position = position
        self.placement_step = step
