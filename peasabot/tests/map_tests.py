import unittest

from peasabot.map_prep import DistanceMap, TargetMap, FreedomMap

from coderone.dungeon.agent import GameState


SAMPLE_BOMB_LOCATIONS = [(0, 4), (2, 6)]
SAMPLE_BLOCKS = [('ib', (0, 6)), ('ib', (1, 3)), ('ib', (2, 0)), ('ib', (2, 3)), ('ib', (3, 2)),
                 ('sb', (0, 1)), ('sb', (1, 1)), ('ob', (1, 0)), ('ob', (3, 3)), ('ob', (3, 4)),
                 ('b', (0, 4)), ('b', (2, 6))]
SAMPLE_PLAYERS = [(0, (1, 5)), (1, (4, 0))]
MAP_SIZE = (5, 7)


class TestMapProcessing(unittest.TestCase):

    def test_distance_map(self):
        game_state = GameState(is_over=False,
                               size=MAP_SIZE,
                               tick_number=0,
                               game_map=[],
                               ammo=[],
                               treasure=[],
                               bombs=SAMPLE_BOMB_LOCATIONS,
                               blocks=SAMPLE_BLOCKS,
                               players=SAMPLE_PLAYERS)

        dm = DistanceMap(MAP_SIZE)

        dm.update(state=game_state, player_pos=(1, 5), player_id=0)

        print(dm._map)
        print(dm.distance_penalty_map)
        print(dm.accessible_area_mask)

        # Check different points
        self.assertAlmostEquals(dm.value_at_point((0, 3)), 13.1)
        self.assertEqual(dm.value_at_point((4, 0)), 0)
        self.assertAlmostEquals(dm.value_at_point((4, 1)), 7.1)
        self.assertAlmostEquals(dm.value_at_point((4, 6)), 4.1)

        # Check unaccessible area
        self.assertAlmostEquals(dm.value_at_point((0, 0)), 0)

    def test_bomb_map(self):
        game_state = GameState(is_over=False,
                               size=MAP_SIZE,
                               tick_number=0,
                               game_map=[],
                               ammo=[],
                               treasure=[],
                               bombs=SAMPLE_BOMB_LOCATIONS,
                               blocks=SAMPLE_BLOCKS,
                               players=SAMPLE_PLAYERS)

        bm = TargetMap(MAP_SIZE)

        bm.update(state=game_state, player_pos=(1, 5), player_id=0)

        self.assertAlmostEquals(bm.value_at_point((0, 0)), 4.0)
        self.assertAlmostEquals(bm.value_at_point((2, 1)), 1.0)
        self.assertAlmostEquals(bm.value_at_point((2, 4)), 1.0)
        self.assertAlmostEquals(bm.value_at_point((1, 2)), 1.0)
        self.assertAlmostEquals(bm.value_at_point((4, 5)), 0.0)
        self.assertAlmostEquals(bm.value_at_point((2, 2)), 0.0)

    def test_freedom_map(self):
        game_state = GameState(is_over=False,
                               size=MAP_SIZE,
                               tick_number=0,
                               game_map=[],
                               ammo=[],
                               treasure=[],
                               bombs=SAMPLE_BOMB_LOCATIONS,
                               blocks=SAMPLE_BLOCKS,
                               players=SAMPLE_PLAYERS)

        fm = FreedomMap(MAP_SIZE)

        fm.update(state=game_state, player_pos=(1, 5), player_id=0)

        print(fm._map)

        self.assertAlmostEquals(fm.value_at_point((0, 0)), 0.0)
        self.assertAlmostEquals(fm.value_at_point((2, 1)), 16.0)
        self.assertAlmostEquals(fm.value_at_point((2, 4)), 16.0)
        self.assertAlmostEquals(fm.value_at_point((1, 2)), 16.0)
        self.assertAlmostEquals(fm.value_at_point((4, 4)), 16.0)
        self.assertAlmostEquals(fm.value_at_point((4, 5)), 49.0)
        self.assertAlmostEquals(fm.value_at_point((4, 6)), 16.0)
        self.assertAlmostEquals(fm.value_at_point((3, 5)), 81.0)
        self.assertAlmostEquals(fm.value_at_point((3, 6)), 16.0)
        self.assertAlmostEquals(fm.value_at_point((2, 2)), 16.0)
