"""

    Our peasant bot.

"""


class agent:
    def __init__(self):
        self.actions = ['', 'u', 'd', 'l', 'r', 'p']
        pass

    def next_move(self, game_state, player_state):
        return self.actions[0]
