"""
The main way for peasants to train.
"""

import argparse
import sys
from typing import List


from coderone.dungeon.agent_driver.multiproc_driver import Driver
from coderone.dungeon.game import Game
from coderone.dungeon.main import __load_or_generate_config, _prepare_import


def _load_agent_drivers(agent_modules, config: dict, watch=False):
    agents = []

    for counter, agent_module in enumerate(agent_modules):
        try:
            module_name = _prepare_import(agent_module)
            driver = Driver(module_name, watch, config)
            agents.append(driver)
        except Exception:
            return None

    return agents


def run_training(agents: List[str], headless: bool, number_of_games: int):
    """ Initialize agents and run a bunch of games. """
    # These values are fixed for the game atm
    row_count = 10
    column_count = 12
    iteration_limit = 1800
    is_interactive = False
    config = __load_or_generate_config(None)
    # Load agent modules
    agents_drivers = _load_agent_drivers(agents, watch=False, config=config)
    if not agents_drivers:
        return None

    agents = [driver.agent() for driver in agents_drivers]

    for i in range(number_of_games):
        game = Game(row_count=row_count, column_count=column_count, max_iterations=iteration_limit)

        # Add all agents to the game
        for agent, driver in zip(agents, agents_drivers):
            game.add_agent(agent, driver.name)

        # Add a player for the user if running in interactive mode or configured interactive
        user_pid = game.add_player("Player") if is_interactive else None

        game.generate_map()

        tick_step = config.get('tick_step')
        if headless:
            from coderone.dungeon.headless_client import Client

            client = Client(game=game, config=config)
            client.run(tick_step)
        else:
            if config.get('hack'):
                from coderone.dungeon.hack_client import Client
                screen_width = 80
                screen_height = 24
            else:
                from coderone.dungeon.arcade_client import Client, WIDTH, HEIGHT, PADDING

                screen_width = PADDING[0] * 2 + WIDTH * 12
                screen_height = PADDING[1] * 3 + HEIGHT * 10

            window = Client(width=screen_width, height=screen_height, title="title", game=game, config=config,
                            interactive=is_interactive, user_pid=user_pid)
            window.run(tick_step)


def main():
    parser = argparse.ArgumentParser(description='*******************')

    parser.add_argument('--headless', action='store_true',
                        default=False,
                        help='run without graphics')

    parser.add_argument('--games', type=int, default=10)

    parser.add_argument("agents", nargs="+", help="agent module")

    args = parser.parse_args()

    if len(args.agents) < 2:
        print("At least 2 agents must be provided in the match mode. Exiting", file=sys.stderr)
        sys.exit(1)

    run_training(args.agents, args.headless, args.games)


if __name__ == "__main__":
    main()
