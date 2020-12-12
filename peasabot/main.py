from coderone.dungeon.main import run_match
import argparse
import sys
import jsonplus


def main():
    parser = argparse.ArgumentParser(description='*******************')

    parser.add_argument('--headless', action='store_true',
                        default=False,
                        help='run without graphics')
    parser.add_argument('--interactive', action='store_true',
                        default=False,
                        help='all a user to contol a player')
    parser.add_argument('--no_text', action='store_true',
                        default=False,
                        help='Graphics bug workaround - disables all text')
    parser.add_argument('--start_paused', action='store_true',
                        default=False,
                        help='Start a game in pause mode, only if interactive')
    parser.add_argument('--players', type=str,
                        help="Comma-separated list of player names")
    parser.add_argument('--hack', action='store_true',
                        default=False,
                        help=argparse.SUPPRESS)

    parser.add_argument('--submit', action='store_true',
                        default=False,
                        help="Don't run the game, but submit the agent as team entry into the trournament")

    parser.add_argument('--record', type=str,
                        help='file name to record game')
    parser.add_argument('--watch', action='store_true',
                        default=False,
                        help='automatically reload agents on file changes')
    parser.add_argument('--config', type=str,
                        default=None,
                        help='path to the custom config file')

    parser.add_argument("agents", nargs="+", help="agent module")

    args = parser.parse_args()

    n_agents = len(args.agents)
    if args.submit:
        if n_agents > 1:
            print(
                "Error: Only a single agent entry per team is allowed.\n"
                f"You have specified {n_agents} agent modules.\n"
                "Please chose only one you wish submit and try again.\n"
                , file=sys.stderr)
            sys.exit(1)

        sys.exit(0)

    if len(args.agents) < 2 and (args.headless or not args.interactive):
        print("At least 2 agents must be provided in the match mode. Exiting", file=sys.stderr)
        sys.exit(1)

    if args.headless and args.interactive:
        print("Interactive play is not support in headless mode. Exiting", file=sys.stderr)
        sys.exit(1)
    if args.headless and args.no_text:
        print("Makes no sense to run headless and ask for no-text. Ignoring", file=sys.stderr)
    if not args.interactive and args.start_paused:
        print("Can not start paused in non-interactive mode. Exiting", file=sys.stderr)
        sys.exit(1)

    jsonplus.prefer_compat()

    players = args.players.split(',') if args.players else None
    result = run_match(agents=args.agents, players=players, config_name=args.config, record_file=args.record,
                       watch=args.watch, args=args)
    print(jsonplus.pretty(result))

    # We done here, all good.
    sys.exit(0)



if __name__ == "__main__":
    main()