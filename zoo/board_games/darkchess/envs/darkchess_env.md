import copy                    # 深拷貝物件
import logging                 # 日誌記錄
import os, sys                 # 系統操作
from functools import lru_cache  # 快取函數結果
from typing import List        # 型別提示

import gymnasium as gym        # OpenAI Gym 環境介面
import numpy as np             # 數值計算
from ding.envs import BaseEnv, BaseEnvTimestep  # DI-engine 環境基類
from ding.utils import ENV_REGISTRY             # 環境註冊器
from easydict import EasyDict                   # 方便的字典類

from zoo.board_games.darkchess.envs.legal_actions_cython import legal_actions_cython  # Cython 加速的合法動作計算

@lru_cache(maxsize=1024)  # 快取最多 1024 個結果，避免重複計算
def _legal_actions_func_lru(board_tuple, player_color):
    board_array = np.array(board_tuple, dtype=np.int32)  # tuple 轉回 numpy
    board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
    return legal_actions_cython(DarkchessEnv.get_all_actions(), board_view, player_color)

@ENV_REGISTRY.register('darkchess')  # 註冊環境，讓框架能用名稱找到
class DarkchessEnv(BaseEnv):

    config = dict(
        env_id="Darkchess",         # 環境名稱
        board_width=4,              # 棋盤寬度（4 列）
        board_height=8,             # 棋盤高度（8 行）
        board_feature_layer=16,     # 特徵層數（16 種棋子狀態）
        battle_mode='self_play_mode',  # 對戰模式
        agent_vs_human=False,       # 是否人機對戰
        stop_value=0.8,             # 訓練停止閾值
        long_catch=3,               # 長捉循環次數（和局判定）
        no_eat_flip=180,            # 無吃翻步數（和局判定）
    )

# 生成所有可能的動作（共 352 個）
# 動作格式：((from_row, from_col), (to_row, to_col))
all_actions = []
board = [
    [(0, 0), (0, 1), (0, 2), (0, 3)],  # 第 8 行
    [(1, 0), (1, 1), (1, 2), (1, 3)],  # 第 7 行
    # ... 共 8 行
    [(7, 0), (7, 1), (7, 2), (7, 3)],  # 第 1 行
]

for i in range(8):
    for j in range(4):
        # 翻棋：起點 = 終點
        all_actions.append((board[i][j], board[i][j]))
        # 垂直移動
        for k in range(8):
            if k != i:
                all_actions.append((board[k][j], board[i][j]))
        # 水平移動
        for k in range(4):
            if k != j:
                all_actions.append((board[i][j], board[i][k]))

# 初始化
def __init__(self, cfg=None):
    self.cfg = cfg
    self.battle_mode = cfg.battle_mode  # 對戰模式
    
    self.board_width = cfg.board_width    # 4
    self.board_height = cfg.board_height  # 8
    self.board_feature_layer = cfg.board_feature_layer  # 16
    
    self.players = [0, 1]           # 玩家編號
    self.total_num_actions = 352    # 動作總數
    
    # 棋子名稱對照
    # 0-6: 紅方（K帥 G仕 M相 R俥 N傌 C炮 P兵）
    # 7-13: 黑方（k將 g士 m象 r車 n馬 c包 p卒）
    # 14: 空格(-), 15: 暗子(X)
    self.chess_name = np.array(['K','G','M','R','N','C','P','k','g','m','r','n','c','p','X','-'])
    
    # 棋子價值（用於判斷誰能吃誰）
    self.chess_value = {0:7, 1:6, 2:5, 3:4, 4:3, 5:2, 6:1, 7:7, 8:6, 9:5, 10:4, 11:3, 12:2, 13:1}
    
    # Gym 空間定義
    self._observation_space = gym.spaces.Box(0, 1, (16, 8, 4), dtype=np.int64)
    self._action_space = gym.spaces.Discrete(352)
    self._reward_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

# 重置遊戲
def reset(self, init_state=None):
    """
    棋盤座標：
    8 |  (0,0) (0,1) (0,2) (0,3)   ← a8 b8 c8 d8
    7 |  (1,0) (1,1) (1,2) (1,3)
    ...
    1 |  (7,0) (7,1) (7,2) (7,3)   ← a1 b1 c1 d1
         ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
           a     b     c     d
    """
    # 初始棋盤全是暗子 (15)
    self.board = np.full((8, 4), 15, dtype=np.int64)
    
    # 棋子數量：紅帥1、紅仕2、紅相2... 共 32 顆
    self.chess_count = np.array([1,2,2,2,2,2,5, 1,2,2,2,2,2,5, 0, 32])
    self.flipped_chess_count = np.array([1,2,2,2,2,2,5, 1,2,2,2,2,2,5])
    
    self.continuous_move_count = 0  # 連續移動計數（無吃翻）
    self.action_history = []        # 歷史動作
    self.player_color = ['U', 'U']  # U=未知, R=紅, B=黑
    self._current_player = 0        # 當前玩家
    
    # 生成合法動作遮罩
    action_mask = np.zeros(352, np.int8)
    action_mask[self.legal_actions] = 1
    
    # 回傳觀察值
    obs = {
        'observation': self.encode_board(),  # (16, 8, 4) 的 one-hot 編碼
        'action_mask': action_mask,          # (352,) 合法動作
        'board': copy.deepcopy(self.board),  # 棋盤狀態
        'current_player_index': self.current_player,
        'to_play': self.current_player,
        'chance': 0                          # 翻棋結果
    }
    return obs
    
# 執行動作
def step(self, action_id: int):
    action = self.all_actions[action_id]  # action_id → tuple
    assert self.is_legal_action(action)   # 確認合法
    
    if self.battle_mode == 'self_play_mode':
        self.action_history.append(action_id)
        timestep = self._player_step(action)
        if timestep.done:
            # 從玩家 1 的視角計算回報
            timestep.info['eval_episode_return'] = (
                -timestep.reward if timestep.obs['to_play'] == 0 else timestep.reward
            )
        return timestep

# 內部執行
def _player_step(self, action: tuple) -> BaseEnvTimestep:
    if action[0] == action[1]:  # 起點 == 終點
        chess_id = self.flip(action)  # 翻棋
        self.chance = chess_id        # 記錄翻到什麼
    else:
        self.move(action)             # 移動/吃子
        self.chance = 0
    
    # 檢查遊戲是否結束
    done, winner = self.get_done_winner()
    reward = np.array(float(winner == self.current_player)).astype(np.float32)
    
    # 換玩家
    self.current_player = self.next_player
    
    # 生成新的觀察值
    action_mask = np.zeros(352, np.int8)
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

# 編碼棋盤
def encode_board(self):
    """
    將棋盤轉成 16 層 one-hot 編碼
    每層代表一種棋子類型，該位置是該棋子則為 1，否則為 0
    輸出形狀：(16, 8, 4)
    """
    state = np.array([_ for _ in range(16)])  # [0, 1, 2, ..., 15]
    layers = state.reshape(16, 1, 1)          # (16, 1, 1)
    obs = (self.board == layers).astype(np.float32)  # 廣播比較
    return obs  # (16, 8, 4)

# 判斷合法
def is_legal_action(self, action) -> bool:
    src, dst = action
    src_chess, dst_chess = self.board[src], self.board[dst]

    if src != dst:  # 移動或吃子
        if self.player_color[self.current_player] == 'U':
            return False  # 顏色未知時只能翻棋
        elif src_chess == 15 or src_chess == 14 or dst_chess == 15:
            return False  # 起點不能是暗子/空格，終點不能是暗子
        elif self.player_color[self.current_player] == 'R':  # 紅方
            if 0 <= src_chess <= 6 and dst_chess == 14 and self.check_neighboring(src, dst):
                return True  # 移動到空格
            elif src_chess == 5:  # 炮
                return self.check_cannon_can_eat(src, dst)  # 特殊判定
            elif src_chess == 0 and dst_chess == 13:  # 帥不能吃卒
                return False
            elif src_chess == 6 and dst_chess == 7:  # 兵可以吃將
                return True
            # ... 更多規則
        # 黑方類似
    elif src_chess != 15:
        return False  # 翻棋的位置必須是暗子
    
    return True

# 移動/吃子
def move(self, action: tuple):
    src, dst = action
    dst_chess = self.board[dst]
    
    if dst_chess != 14:  # 目標不是空格 = 吃子
        chess_id = np.where(self.chess_name == dst_chess)[0]
        self.chess_count[chess_id] -= 1  # 減少被吃棋子的數量
        self.continuous_move_count = 0   # 重置計數
    else:
        self.continuous_move_count += 1  # 純移動，計數 +1
    
    self.board[dst] = self.board[src]    # 目標位置 = 來源棋子
    self.board[src] = 14                 # 來源位置 = 空格

# 翻棋
def flip(self, action: tuple):
    chess_id = self.get_random_chess_id()  # 隨機選一個暗子
    
    # 第一次翻棋決定雙方顏色
    if self.player_color[self.current_player] == 'U':
        if chess_id <= 6:  # 翻到紅棋
            self.player_color[self.current_player] = 'R'
            self.player_color[self.next_player] = 'B'
        else:  # 翻到黑棋
            self.player_color[self.current_player] = 'B'
            self.player_color[self.next_player] = 'R'
    
    self.chess_count[15] -= 1              # 暗子數量 -1
    self.flipped_chess_count[chess_id] -= 1  # 該棋子剩餘數量 -1
    self.board[action[0]] = chess_id       # 棋盤上顯示翻開的棋子
    self.continuous_move_count = 0         # 重置計數
    
    return chess_id  # 回傳翻到的棋子 ID

# 判斷勝負
def get_done_winner(self):
    # 對手無步可走 → 當前玩家勝
    self.current_player = self.next_player
    if len(self.legal_actions) == 0:
        self.current_player = self.next_player
        return True, self.current_player
    else:
        self.current_player = self.next_player
    
    # 超過 180 步無吃翻 → 和局
    if self.continuous_move_count >= self.no_eat_flip:
        return True, -1
    
    # 長捉（同樣動作循環 3 次）→ 和局
    if self.continuous_move_count >= self.long_catch * 4:
        recent_actions = np.array(self.action_history[-self.long_catch * 4:])
        cycles = recent_actions.reshape(self.long_catch, 4)
        if np.all(cycles == cycles[0], axis=1).all():
            return True, -1
    
    return False, -1  # 遊戲繼續

# 轉換格式
def action_to_string(self, action_id: int):
    """將 action_id 轉成 MGTP 格式"""
    action = self.all_actions[action_id]
    # (row, col) → "a8" 格式
    src = chr(ord('a') + action[0][1]) + str(8 - action[0][0])
    dst = chr(ord('a') + action[1][1]) + str(8 - action[1][0])
    return src + '-' + dst  # 例如: "a8-b8"

|函數|用途|
|----|----|
|'reset()'|	重置遊戲|
|'step(action_id)'|	執行動作|
|'encode_board()'|	棋盤 → 神經網路輸入|
|'is_legal_action()'|	判斷動作是否合法|
|'move()'|	執行移動/吃子|
|'flip()'|	執行翻棋|
|'get_done_winner()'|	判斷遊戲結束與勝負|
|'action_to_string()'|	action_id → MGTP 字串|