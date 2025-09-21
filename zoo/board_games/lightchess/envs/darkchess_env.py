import copy
from typing import List

# import gymnasium as gym
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.atari.envs.atari_wrappers import wrap_lightzero


@ENV_REGISTRY.register('darkchess')
class DarkchessEnv(BaseEnv):
    config = dict(
        env_id="DarkChess",
        board_width=4,
        board_height=8,
        no_eat_flip=180,
        long_catch=3,
        total_num_actions=352,
        battle_mode='play_with_bot_mode',
        # (str) The mode of the environment when doing the MCTS.
        battle_mode_in_simulation_env='self_play_mode',  # only used in AlphaZero
        # (str) The render mode. Options are 'None', 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.
        # If None, then the game will not be rendered.
        render_mode=None,
        # (bool) Whether to let human to play with the agent when evaluating. If False, then use the bot to evaluate the agent.
        agent_vs_human=False,
        # (float) The stop value when training the agent. If the evalue return reach the stop value, then the training will stop.
        stop_value=2,
        # (bool) Whether to use the MCTS ctree in AlphaZero. If True, then the AlphaZero MCTS ctree will be used.
        alphazero_mcts_ctree=False,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg
        self.board_width = cfg.board_width
        self.board_height = cfg.board_height
        self.long_catch = cfg.long_catch
        self.no_eat_flip = cfg.no_eat_flip
        self.battle_mode = cfg.battle_mode
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # The mode of MCTS is only used in AlphaZero.
        self.battle_mode_in_simulation_env = 'self_play_mode'
        # options = {None, 'state_realtime_mode', 'image_realtime_mode', 'image_savefile_mode'}
        self.render_mode = cfg.render_mode
        assert self.render_mode in [None, 'state_realtime_mode', 'image_realtime_mode', 'image_savefile_mode']
        self.players = [1, 2]
        self._current_player = 1
        self.total_num_actions = self.total_num_actions
        self._env = self

    def reset(self, first_player=True) -> dict:
        # Reset board to all flipped
        """
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
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)

        self.board = np.char.chararray((self.board_height, self.board_width))
        self.board[:] = 'X'

        # The remaining number of all pieces
        # 0 ~ 6: 帥 (K)、仕 (G)、相 (M)、俥 (R)、傌 (N)、炮 (C)、兵 (P)
        # 7 ~ 13: 將 (k)、士 (g)、象 (m)、車 (r)、馬 (n)、包 (c)、卒 (p)
        # 14: 空格 (-) 15: 暗子 (X)
        self.chess_count = np.array([1, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 5, 0, 32])
        self.flipped_chess_count = np.array([1, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 5])

        self.continuous_move_count = 0

        self._observation_space = gym.spaces.Box(0, 15, (self.board_width, self.board_height, 16), dtype=int)
        self._action_space = gym.spaces.Discrete(self.total_num_actions)
        self._reward_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        obs = self._env.reset()
        self.obs = to_ndarray(obs)
        obs = self.observe()
        return obs

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def observe(self) -> dict:
        observation = self.obs

        action_mask = np.ones(self._action_space.n, 'int8')
        return {'observation': observation, 'action_mask': action_mask, 'to_play': -1}
