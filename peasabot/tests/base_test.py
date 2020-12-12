
import unittest

import importlib

import peasabot


class TestAgentImport(unittest.TestCase):

    def test_import(self):
        b = importlib.import_module('/home/gimait/programs/DaDSbot/peasabot')
        self.assertEqual(b.agent(), peasabot.agent())
