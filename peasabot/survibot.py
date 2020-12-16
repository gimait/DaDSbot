"""
Our peasant bot.
"""
from coderone.dungeon.agent import GameState, PlayerState

from .consumer_bot import ConsumerBot
import time

MAX_BOMB = 5  # Don't pick up more
MIN_BOMB = 1  # Don't place bomb
BOMB_TICK_THRESHOLD = 15  # Added time to block the tile for future bombs --> Planner
CORNER_THRESH = 30  # Threshold that indicates when a spot has a very low degree of freedom
ATTACK_THRESH = 70
DANGER_THRESH = 0  # <-- NOT USED
RUNAWAY_TILES = 2

DEBUG = 1
class Agent(ConsumerBot):
    """ Agent bot."""
    def __init__(self):
        super().__init__((12, 10), BOMB_TICK_THRESHOLD)

    def next_move(self, game_state: GameState, player_state: PlayerState):
        t0 = time.perf_counter()

        full_map = self.print_map(game_state)
        updated_map = (self.full_map_prev == full_map if self.full_map_prev is not None else True)
        # Yes I debug with a print(updated_map)
        if not self.planned_actions:
            self.update_state(game_state, player_state)
            self.planned_actions = self._next_move_survibot()
        elif updated_map and self.keep_plan <= 0:
            self.update_state(game_state, player_state)
            self.planned_actions = self._next_move_survibot()
        else:  # Map is not updated and there are planned actions
            if DEBUG:
                print('pass ' + str(updated_map))
            pass

        self.keep_plan = self.keep_plan - 1 # Variable to ignore rest and RUNAWAY the number of tiles of the VARIABLE
        action = (self.planned_actions.pop(0) if self.planned_actions else '')
        if action == 'p':
            self.previous_plan = 'run'

        #    if self.free_map._map[self.location] < CORNER_THRESH and self.bomb_management_map.all_map[self.location] < 1:
        #        if not self.planned_actions:
        #            action = ''
        #        else:
        #            action = self.planned_actions.pop(0)
        #    self.previous_plan = 'run'

        self.full_map_prev = full_map  # Update the map for checking if change in the next tick
        self.last_move = action

        dt = time.perf_counter() - t0
        if dt > 0.05:
            print("Cuidao!! {}".format(dt))
        return action

    def _next_move_survibot(self):
        # Agent possibilities
        danger_zone, danger_status = self.bomb_management_map.is_in_danger()
        ammo_tile, ammo_status = self.is_ammo_avail()

        treasure_tile, treasure_status = self.is_treasure_avail()
        ore_tile, ore_status = self.is_ore_hot()    # <- I assume ore is hot if only one bomb is left

        kill_tiles, kill_status = self.is_killing_an_option()

        plan= ['']
        # TANGENTIAL Behaviours Top priorities.
        if danger_status:
            if DEBUG:
                print('DANGER status')
            plan, _ = self.plan_to_safest_area(danger_zone)
        elif self.previous_plan == "run":
            d = danger_zone if self.bomb_management_map.last_placed_bomb is None \
                else danger_zone - self.bomb_management_map.last_placed_bomb._map
            plan, _ = self.path_to_freest_area(d) # <- change for freest area which multiplies for the accesible_area_mask
            self.previous_plan = None
            self.keep_plan = RUNAWAY_TILES
            if DEBUG:
                print('RUN '+ str(plan))
        elif ammo_status and self.ammo < MAX_BOMB:
            plan, _ = self.plan_to_tile(ammo_tile)
            if DEBUG:
                print('AMMO I GO '+ str(plan))
        else: # FARM
            # 1  point to pick up is always good
            if treasure_status:
                plan, _ = self.plan_to_tile(treasure_tile)
                if DEBUG:
                    print('FARM - TREASURE  ' + str(plan))
            elif self.ammo > MIN_BOMB:
                # 2 Farm a hot ORE -> One bomb left
                if ore_status:
                    plan, connected = self.plan_to_tile(ore_tile)
                    if connected:
                        plan.append('p')
                    if DEBUG:
                        print('FARM - ORE  ' + str(plan))

                # 3 Place a bomb in a good place if you have bombs
                elif self.previous_plan == "loot" or self.ammo > MIN_BOMB:
                    best_point_for_bomb = self.get_best_point_for_bomb()
                    plan, connected = self.plan_to_tile(best_point_for_bomb)
                    self.previous_plan = (None if not plan else "loot")
                    if connected:
                        plan.append('p')
                    if DEBUG:
                        print('FARM - LOOT  ' + str(plan))

            # If there is still ammo around and we are bored, let's go catch it
            # elif ammo_status:
            #    plan, _ = self.plan_to_tile(ammo_tile)


        #        # KILL
#        elif (0 < self.free_map._map[self.opponent_tile] < ATTACK_THRESH) and \
#             (self.previous_plan == "kill" or kill_status):
#            plan, connected = self.plan_to_tile(kill_tiles)
#            self.previous_plan = (None if not plan else "kill")
#            if connected:
#                plan.append('p')

  #      # 4 If we finished a plan, get away
  #      elif self.previous_plan == "run":
  #          d = danger_zone if self.bomb_management_map.last_placed_bomb is None \
  #              else danger_zone - self.bomb_management_map.last_placed_bomb._map
  #          plan, _ = self.path_to_freest_area(d) # <- change for freest area which multiplies for the accesible_area_mask
  #          self.previous_plan = None
        # MOVE FREE-SAFE SPACE
  #      else:
  #          free_tile = self.get_freedom_tiles()
  #          plan, _ = self.plan_to_tile(free_tile)
        if DEBUG:
            print('*' * 25 )
        return plan


    def reset(self):
        pass
