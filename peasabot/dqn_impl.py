"""

Implementation for Deep Q Learning

"""

import math
import os
import random
import time
from typing import Tuple

import numpy as np

import torch

from .memory_buffer import MemoryBuffer, ShortTermMemoryBuffer
from coderone.dungeon.agent import GameState, PlayerState
# from DaDSbot.misc.plots import plot_graph


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "agent_model.pt"

BATCH_SIZE = 256
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
GAMMA = 0.9
TARGET_UPDATE = 10


class GameMemoryBuffer(MemoryBuffer):
    """ Memory buffer for action replay. """
    def get(self, batch_size: int) -> Tuple[np.array, np.array, np.array]:
        data = self.rand_sample(batch_size)
        samp = []
        next_state = []
        out = []
        action = []
        for step in data:
            samp.append(torch.from_numpy(step['map']).float())
            if 'next' in step:
                next_state.append(torch.from_numpy(step['next']).float())
            else:
                next_state.append(None)
            out.append(torch.FloatTensor([step.get('score')]))
            action.append(step['action'])
        return samp, next_state, out, action


class AgentNN(torch.nn.Module):
    """ Simple network. """
    def __init__(self, D_in, D_out):
        super(AgentNN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 20)
        self.h1 = torch.nn.Linear(20, 15)
        self.linear2 = torch.nn.Linear(15, D_out)
        self.activation = torch.nn.Tanh()

    def forward(self, x, mask=None):
        x = self.activation(self.linear1(x.float()))
        x = self.activation(self.h1(x))
        y_pred = self.activation(self.linear2(x))

        if mask is not None:
            y_pred[mask == 0] = 0

        return y_pred

    def save_model(self, path: str = None) -> None:
        if not path:
            path = "agent_model.pt"
        torch.save(self.state_dict(), path)


class GameQLearner:
    """ Learning algorithm. """
    def __init__(self, player_num, plotting=False):
        self.name = "aibot"
        self.player_num = player_num - 1
        # self.env = env
        # self.plotting = plotting
        ######################
        # Network definition
        # Agent in -> current state = map (5x7) + bombs (2 x 2) + players (2 x(pos + prev pos + score) = 10) + turn
        # Agent outD_in -> expected scores for next possible actions (move, stay or place bomb = 6)
        ACTIONS_DICT = ['', 'u', 'd', 'l', 'r', 'p']
        cols, rows = 10, 10
        self.stm_buffer_size = 1
        # self.n_bombs = 2
        self.target_net = AgentNN((cols * rows) * self.stm_buffer_size,
                                  len(ACTIONS_DICT)).to(device)
        self.policy_net = AgentNN((cols * rows) * self.stm_buffer_size,
                                  len(ACTIONS_DICT)).to(device)

        print("AMUNICIONNNNNNN")
        if os.path.isfile(self.name + ".pt"):
            self.policy_net.load_state_dict(torch.load(self.name + ".pt"))
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_net.eval()

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())

        # Memory
        self.memory = GameMemoryBuffer(int(1000000 / self.stm_buffer_size))
        self.stm = ShortTermMemoryBuffer(self.stm_buffer_size)

        # Play variables
        self.first_iteration = True
        self.last_score = 0

        # Show off players:
        self.players = []

        # Training variables
        self.batch_size = BATCH_SIZE
        self.steps_done = 0
        self.game_steps = []
        self.last_action = None
        self.n_turns = 0

        # Misc variables
        self.last_time = time.perf_counter()
        self.game = 0
        self.idle_steps = 0
        self.bombs_placed = 0
        self.avg_score = [0, 0, 0]
        self.running_loss = 0
        self.all_bombs = []
        self.all_idles = []
        self.all_scores = []
        self.all_turns = []
        self.all_loss = []
        self.game_idx = []

    def give_next_move(self, game_state: GameState, player_state: PlayerState):
        '''
        This method is called each time the player needs to choose an
        actionlearning_thread.start()
        solid_state: is a dictionary containing all the information about the board
        '''

        self.board = game_state.bombs   #todo
        self.done = game_state.is_over
        self.bombs = game_state.bombs
        # self.turn = game_state.
        self.player = player_state
        enemy = game_state.opponents

        # In first iteration, initialize stm
        full_state = np.concatenate((np.array(self.player.position),
                                     np.array(self.player.prev_position),
                                     np.array(enemy),
                                     np.array(enemy.prev_position),
                                     self.bomb_to_timer_array(self.bombs),
                                     np.asarray(self.board).reshape(-1)))

        if self.first_iteration:
            self.first_iteration = False
            for i in range(self.stm_buffer_size):
                self.stm.append(full_state)
        else:
            self.stm.append(full_state)

        self._apply_map_and_scores(self.player.score - self.last_score)
        self.last_score = self.player.score

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)

        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.from_numpy(self.stm.get()).float().to(device)
                pred = self.policy_net(state[None, :]).data.cpu().numpy()
                action = int(np.argmax(pred))
                if action == 0:
                    self.idle_steps += 1
                if action == 5:
                    self.bombs_placed += 1
        else:
            player = random.choice(self.players)
            # action = player.give_next_move(solid_state)

        #############################
        self.last_action = action
        self.n_turns += 1

        return action

    # def bomb_to_timer_array(self, bombs):
    #     out = np.array([-1 for _ in range(self.n_bombs)])
    #     for b in bombs:
    #         out[b.owned_by] = b.timer
    #     return out

    def is_done(self, board, players):
        self.board = board
        self.player = players[self.player_num]
        self.game += 1
        self.steps_done += 1
        # If we lost, do like a bomb hit you. Failure is not accepted in Kobra Kai
        if players[self.player_num].score == min([p.score for p in players]):
            self.player.score -= 40
        self._apply_map_and_scores(self.player.score - self.last_score)
        # self._process_scores_after_match()
        for s in self.game_steps[:-1]:
            self.memory.append(s)

        if self.game % 1000 == 0 and self.plotting:
            print("score 1: {:.2f}, \tscore 2: {:.2f}, \tturns: {:.2f}, \tgame: {}, \tt: {:.2f}, \tloss: {}"
                  .format(self.avg_score[0],
                          self.avg_score[1],
                          self.avg_score[2],
                          self.game,
                          time.perf_counter() - self.last_time,
                          self.running_loss))
            self.last_time = time.perf_counter()
            self.all_bombs += [self.bombs_placed]
            self.all_idles += [self.idle_steps]
            self.all_scores += [self.avg_score[0]]
            self.all_turns += [self.avg_score[2]]
            self.all_loss += [self.running_loss]
            self.game_idx += [self.game]

            # plot_graph(self.game_idx, self.all_scores, self.all_bombs, self.all_idles)

            if self.game % 1000000 == 0:
                torch.save(self.target_net.state_dict(), 'model.weights')

            self.avg_score = [0, 0, 0]
            self.running_loss = 0
            self.bombs_placed = 0
            self.idle_steps = 0
        else:
            self.avg_score[0] += players[0].score / 1000.0
            self.avg_score[1] += players[1].score / 1000.0
            self.avg_score[2] += self.n_turns / 1000.0

        self._learning_step()

        if self.game % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # if self.game % 10000 == 0:
        #     self.target_net.save_model(self.name + ".pt")

        self.reset()

    def reset(self):
        self.game_steps = []
        self.last_action = None
        self.first_iteration = True
        self.n_turns = 0
        self.last_score = 0

    # Learning methods:
    def _learning_step(self):
        if len(self.memory) < self.batch_size:
            return

        # First, get a batch sample from memory.
        # For each element of the batch, we get a current state plus the action taken in that step, what reward was
        # obtained and the next state.
        state, next_state, action_rwd, action = self.memory.get(self.batch_size)

        # Create a mask and index of non-final states. This is used to identify when a future state cannot give any
        # reward (but the current state can still).
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                      next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in next_state
                                             if s is not None], dim=0).to(device)

        # Create the batch objects for training
        state_batch = torch.stack(state).to(device)
        reward_batch = torch.cat(action_rwd).float().to(device)
        action_batch = torch.tensor([action], device=device, dtype=torch.long).permute(1, 0)

        # Now, make the network predict a result for the future states
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # And add it to the known recieved reward
        Q_values = reward_batch + GAMMA * next_state_values

        # The prediction for the current state is what we compare with the Q_values:
        state_action_values = self.policy_net(state_batch).gather(0, action_batch)

        # Compute Huber loss
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, Q_values.unsqueeze(1))
        self.running_loss += loss.item()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # stopping the clipping of the neurons, so we can generalize the output to any range of values.
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _apply_map_and_scores(self, score):
        # Store the current map
        self.game_steps.append({'map': self.stm.get(), 'score': 0})

        if len(self.game_steps) > 1:
            # Store the last action, reward and next map state
            self.game_steps[-2]['action'] = self.last_action
            self.game_steps[-2]['next'] = self.game_steps[-1]['map']
            self.game_steps[-2]['score'] = self._transform_reward(score)

    @staticmethod
    def _transform_reward(reward):
        # Reward value ranges from -45 (bombed) to 30 (explode 3 blocks), so we'll map it to a range
        # (-45, 45) -> (-1, 1)
        return reward / 45.0

    def _process_scores_after_match(self):
        for i, step in enumerate(self.game_steps[:-1]):
            for j in range(1, len(self.game_steps) - 1 - i):
                step['score'] += (self.game_steps[i + j]['score'] * 0.8 ** j
                                  if self.game_steps[i + j]['score'] > 0 else 0)
