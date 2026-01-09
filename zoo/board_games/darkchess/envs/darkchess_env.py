import copy
import logging
import os
import sys
from functools import lru_cache
from typing import List

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

# from zoo.board_games.darkchess.envs.get_done_winner_cython import get_done_winner_cython
from zoo.board_games.darkchess.envs.legal_actions_cython import legal_actions_cython


@lru_cache(maxsize=1024)
def _legal_actions_func_lru(board_tuple, player_color):
    # Convert tuple to NumPy array.
    board_array = np.array(board_tuple, dtype=np.int32)
    # Convert NumPy array to memory view.
    board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
    return legal_actions_cython(DarkchessEnv.get_all_actions(), board_view, player_color)


# @lru_cache(maxsize=512)
# def _get_done_winner_func_lru(board_size, board_tuple):
#     # Convert tuple to NumPy array.
#     board_array = np.array(board_tuple, dtype=np.int32)
#     # Convert NumPy array to memory view.
#     board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
#     return get_done_winner_cython(board_size, board_view)

@ENV_REGISTRY.register('darkchess')
class DarkchessEnv(BaseEnv):

    config = dict(
        # (str) The name of the environment registered in the environment registry.
        env_id="Darkchess",
        # (int) The width of the board.
        board_width=4,
        # (int) The height of the board.
        board_height=8,
        # (int) The layer of the board feature.
        board_feature_layer=16,
        # (str) The mode of the environment when take a step.
        battle_mode='self_play_mode',
        # (bool) Whether to use the 'channel last' format for the observation space. If False, 'channel first' format is used.
        # channel_last=False,
        # (bool) Whether to let human to play with the agent when evaluating. If False, then use the bot to evaluate the agent.
        agent_vs_human=False,
        # (float) The stop value when training the agent. If the evalue return reach the stop value, then the training will stop.
        stop_value=0.8,
        # (bool) Whether to use the MCTS ctree in AlphaZero. If True, then the AlphaZero MCTS ctree will be used.
        # alphazero_mcts_ctree=False,
        # (int) The number of long catch cycle needed to draw.
        long_catch=3,
        # (int) The number of no-eat-flip moves needed to draw.
        no_eat_flip=180,
    )

    # Generate all actions
    all_actions = []
    board = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(3, 0), (3, 1), (3, 2), (3, 3)],
        [(4, 0), (4, 1), (4, 2), (4, 3)],
        [(5, 0), (5, 1), (5, 2), (5, 3)],
        [(6, 0), (6, 1), (6, 2), (6, 3)],
        [(7, 0), (7, 1), (7, 2), (7, 3)],
    ]

    for i in range(8):
        for j in range(4):
            all_actions.append((board[i][j], board[i][j]))
            for k in range(8):
                if k != i:
                    all_actions.append((board[k][j], board[i][j]))
            for k in range(4):
                if k != j:
                    all_actions.append((board[i][j], board[i][k]))

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    
    @classmethod
    def get_all_actions(cls: type):
        return cls.all_actions

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._init_flag = False  # TODO: needed?
        self.battle_mode = cfg.battle_mode
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']

        self.board_width = cfg.board_width
        self.board_height = cfg.board_height
        self.board_feature_layer = cfg.board_feature_layer
        self.agent_vs_human = cfg.agent_vs_human
        self.long_catch = cfg.long_catch
        self.no_eat_flip = cfg.no_eat_flip

        # 0 = Player 1, 1 = Player 2
        self.players = [0, 1]
        # 32 flip + 320 move
        self.total_num_actions = 352
        self.chess_name = np.array(['K', 'G', 'M', 'R', 'N', 'C', 'P', 'k', 'g', 'm', 'r', 'n', 'c', 'p', 'X', '-'])
        self.chance = 0

        self.chess_value = {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 7, 8: 6, 9: 5, 10: 4, 11: 3, 12: 2, 13: 1}

        # gym space
        self._observation_space = gym.spaces.Box(
            0, 1, (self.board_feature_layer, self.board_height, self.board_width), dtype=np.int64
        )
        self._action_space = gym.spaces.Discrete(self.total_num_actions)
        self._reward_space = gym.spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32)

        self._env = self

    def reset(self, init_state=None):
        """
        Reset board to all flipped.
        Board:
        8 |  3  2  1  0
        7 |  7  6  5  4
        6 | 11 10  9  8
        5 | 15 14 13 12
        4 | 19 18 17 16
        3 | 23 22 21 20
        2 | 27 26 25 24
        1 | 31 30 29 28
           ‾‾‾‾‾‾‾‾‾‾‾‾
             a  b  c  d
        """
        if init_state is not None:
            self.board = np.array(copy.deepcopy(init_state), dtype=np.int64)
        else:
            self.board = np.full((self.board_height, self.board_width), 15, dtype=np.int64)

        """
        The remaining number of all pieces.
        0 ~ 6: 帥 (K)、仕 (G)、相 (M)、俥 (R)、傌 (N)、炮 (C)、兵 (P)
        7 ~ 13: 將 (k)、士 (g)、象 (m)、車 (r)、馬 (n)、包 (c)、卒 (p)
        14: 空格 (-) 15: 暗子 (X)
        """
        self.chess_count = np.array([1, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 5, 0, 32])
        self.flipped_chess_count = np.array([1, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 5])
        self.continuous_move_count = 0
        self.action_history = []
        # U = unknown, R = red, B = black
        self.player_color = ['U', 'U']
        self._current_player = 0
        # Chance outcome
        self.chance = 0

        action_mask = np.zeros(self.total_num_actions, np.int8)
        action_mask[self.legal_actions] = 1

        # In ``play_with_bot_mode`` and ``eval_mode``, we need to set the "to_play" parameter in the "obs" dict to -1,
        # because we don't take into account the alternation between players.
        # The "to_play" parameter is used in the MCTS algorithm.
        obs = {
            'observation': self.encode_board(),
            'action_mask': action_mask,      
            'board': copy.deepcopy(self.board),
            'current_player_index': self.current_player,
            'to_play': self.current_player if self.battle_mode == 'self_play_mode' else -1,
            'chance': 0
        }

        return obs

    def step(self, action_id: int):
        action = self.all_actions[action_id]
        assert (self.is_legal_action(action))

        if self.battle_mode == 'self_play_mode':
            self.action_history.append(action_id)
            timestep = self._player_step(action)
            if timestep.done:
                # The eval_episode_return is calculated from Player 1's perspective.
                timestep.info['eval_episode_return'
                              ] = -timestep.reward if timestep.obs['to_play'] == 0 else timestep.reward
            return timestep
        # TODO
        # elif self.battle_mode == 'play_with_bot_mode':
        #     timestep_player1 = self._player_step(action)

    def _player_step(self, action: tuple) -> BaseEnvTimestep:
        if action[0] == action[1]:
            chess_id = self.flip(action)
            self.chance = chess_id
        else:
            self.move(action)
            self.chance = 0
        if not 0 <= self.chance < 14:
            logging.warning(f"Chance value: {self.chance}!!!")

        # check if game is end and get winner
        done, winner = self.get_done_winner()
        reward = np.array(float(winner == self.current_player)).astype(np.float32)
        info = {'next player to play': self.next_player}
        # change player
        self.current_player = self.next_player

        if done:
            info['eval_episode_return'] = reward
        action_mask = np.zeros(self.total_num_actions, np.int8)
        action_mask[self.legal_actions] = 1
        obs = {
            'observation': self.encode_board(),
            'action_mask': action_mask,
            'board': copy.deepcopy(self.board),
            'current_player_index': self.current_player,
            'to_play': self.current_player,
            'chance': self.chance
        }
        return BaseEnvTimestep(obs, reward, done, info)

    def encode_board(self):
        # Each layer stands for one state, if fits then marked as 1, otherwise 0.
        # Use broadcasting in numpy to be more efficiency.
        state = np.array([_ for _ in range(16)])
        layers = state.reshape(16, 1, 1)
        obs = (self.board == layers).astype(np.float32)
        # (C, H, W)
        return obs

    def is_legal_action(self, action) -> bool:
        src, dst = action
        src_chess, dst_chess = self.board[src], self.board[dst]

        if src != dst:  # 移動或吃子
            if self.player_color[self.current_player] == 'U':  # 雙方顏色未知時只能翻棋
                return False
            elif src_chess == 15 or src_chess == 14 or dst_chess == 15:  # 起點/終點不能是暗子且起點不能是空棋
                return False
            elif self.player_color[self.current_player] == 'R':
                if 0 <= src_chess <= 6 and dst_chess == 14 and self.check_neighboring(src, dst):
                    return True  # 終點是空格可以直接移動
                elif 7 <= src_chess <= 13 or 0 <= dst_chess <= 6:
                    return False
                elif src_chess == 5:  # (C)
                    return self.check_cannon_can_eat(src, dst)  # 炮要特殊判定
                elif not self.check_neighboring(src, dst):
                    return False
                elif src_chess == 0 and dst_chess == 13:  # (K) (p)
                    return False  # 帥不能吃卒
                elif src_chess == 6 and dst_chess == 7:
                    return True  # 兵可以吃將
                elif self.chess_value[src_chess] < self.chess_value[dst_chess]:
                    return False
            elif self.player_color[self.current_player] == 'B':
                if 7 <= src_chess <= 13 and dst_chess == 14 and self.check_neighboring(src, dst):
                    return True  # 終點是空格可以直接移動
                elif 0 <= src_chess <= 6 or 7 <= dst_chess <= 13:
                    return False
                elif src_chess == 12:  # (c)
                    return self.check_cannon_can_eat(src, dst)  # 包要特殊判定
                elif not self.check_neighboring(src, dst):
                    return False
                elif src_chess == 7 and dst_chess == 6:  # (k) (P)
                    return False  # 將不能吃兵
                elif src_chess == 13 and dst_chess == 0:
                    return True  # 卒可以吃帥
                elif self.chess_value[src_chess] < self.chess_value[dst_chess]:
                    return False
        elif src_chess != 15:
            return False  # 要翻開的那格只能是暗子

        return True  # 剩餘的皆為為合法 action

    def check_neighboring(self, src: tuple, dst: tuple) -> bool:
        if (src[0] == dst[0] and abs(src[1] - dst[1]) == 1):
            return True
        elif (src[1] == dst[1] and abs(src[0] - dst[0]) == 1):
            return True
        return False

    def check_cannon_can_eat(self, src: tuple, dst: tuple) -> bool:
        # 兩顆棋之間有多少棋
        chess_cnt = 0

        # 由於前面已經判定過移動到相鄰空格，因此這裡只會是距離一格以上的空格
        if self.board[dst] == 14:
            return False
        # 炮/包必須隔著一顆棋吃
        if self.check_neighboring(src, dst):
            return False

        if src[0] == dst[0]:  # 兩顆棋在同一個 row
            if src[1] < dst[1]:
                chess_cnt = np.count_nonzero(self.board[src[0], (src[1] + 1):dst[1]] != 14)
            else:
                chess_cnt = np.count_nonzero(self.board[src[0], (dst[1] + 1):src[1]] != 14)
        else:  # 兩顆棋在同一個 column
            if src[0] < dst[0]:
                chess_cnt = np.count_nonzero(self.board[(src[0] + 1):dst[0], src[1]] != 14)
            else:
                chess_cnt = np.count_nonzero(self.board[(dst[0] + 1):src[0], src[1]] != 14)

        return chess_cnt == 1

    def get_random_chess_id(self) -> int:
        # 從剩餘的暗子隨機挑一個
        rand = np.random.randint(self.chess_count[15])
        random_chess_id = -1
        for i in range(14):
            rand -= self.flipped_chess_count[i]
            if rand < 0:
                random_chess_id = i
                break
        assert 0 <= random_chess_id < 14
        return random_chess_id

    def move(self, action: tuple):
        src, dst = action
        dst_chess = self.board[dst]
        if dst_chess != 14:  # 吃子
            chess_id = np.where(self.chess_name == dst_chess)[0]
            self.chess_count[chess_id] -= 1
            self.continuous_move_count = 0
        else:
            self.continuous_move_count += 1
        self.board[dst] = self.board[src]
        self.board[src] = 14

    def flip(self, action: tuple):
        chess_id = self.get_random_chess_id()

        # 第一次翻棋後決定雙方顏色
        if self.player_color[self.current_player] == 'U':
            if chess_id <= 6:
                self.player_color[self.current_player] = 'R'
                self.player_color[self.next_player] = 'B'
            else:
                self.player_color[self.current_player] = 'B'
                self.player_color[self.next_player] = 'R'

        self.chess_count[15] -= 1
        self.flipped_chess_count[chess_id] -= 1
        self.board[action[0]] = chess_id
        self.continuous_move_count = 0

        return chess_id

    def get_done_winner(self):
        # TODO: get done winner
        # 若對手無步可走，則當前玩家獲勝
        self.current_player = self.next_player
        if len(self.legal_actions) == 0:
            self.current_player = self.next_player
            return True, self.current_player
        else:
            self.current_player = self.next_player

        # 超過指定步數無吃翻
        if self.continuous_move_count >= self.no_eat_flip:
            return True, -1  # 平手

        # 長捉（4 步一循環）
        if self.continuous_move_count >= self.long_catch * 4:
            recent_actions = np.array(self.action_history[-self.long_catch * 4:])
            cycles = recent_actions.reshape(self.long_catch, 4)
            if not np.all(cycles == cycles[0], axis=1).all():
                return False, -1
            return True, -1
        return False, -1

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def render(self):
        pass

    def close(self) -> None:
        pass

    def clone(self):
        return copy.deepcopy(self)

    def action_to_string(self, action_id: int):
        # TODO: chess_id to char
        action = self.all_actions[action_id]
        src = chr(ord('a') + action[0][1]) + str(8 - action[0][0])
        dst = chr(ord('a') + action[1][1]) + str(8 - action[1][0])
        return src + '-' + dst

    def show_board(self):
        # TODO: chess_id to char
        for i in range(8):
            for j in range(4):
                print(self.board[i, j], end=' ')
            print()

    @property
    def current_player(self):
        return self._current_player

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    @property
    def next_player(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @property
    def legal_actions(self):
        # Convert NumPy arrays to nested tuples to make them hashable.
        return _legal_actions_func_lru(tuple(map(tuple, self.board)), self.player_color[self.current_player])
    
    # @property
    # def legal_actions(self):
    #     legal_actions = []
    #     for id, action in enumerate(self.all_actions):
    #         if (self.is_legal_action(action)):
    #             legal_actions.append(id)
    #     return legal_actions

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # In eval phase, we use ``eval_mode`` to make agent play with the built-in bot to
        # evaluate the performance of the current agent.
        # cfg.battle_mode = 'eval_mode'
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "LightZero Darkchess Env"
