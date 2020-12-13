"""
Utility functions
"""


def get_opponents(pid, players):
    return [_pos for (_id, _pos) in players if pid != _id]
