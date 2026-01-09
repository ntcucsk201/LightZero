import sys
import torch
from lzero.policy.muzero import MuZeroPolicy
from zoo.board_games.darkchess.config.muzero_darkchess_config import main_config, create_config
from zoo.board_games.darkchess.envs.darkchess_env import DarkchessEnv

# MGTP 命令列表
COMMANDS = [
    "protocol_version",  # 0
    "name",              # 1
    "version",           # 2
    "known_command",     # 3
    "list_commands",     # 4
    "quit",              # 5
    "boardsize",         # 6
    "reset_board",       # 7
    "num_repetition",    # 8
    "num_moves_to_draw", # 9
    "move",              # 10
    "flip",              # 11
    "genmove",           # 12
    "game_over",         # 13
    "ready",             # 14
    "time_settings",     # 15
    "time_left",         # 16
    "showboard",         # 17
    "init_board",        # 18
]

def pos_to_coord(pos: str) -> tuple:
    """將 MGTP 座標轉成 (row, col)"""
    col = ord(pos[0]) - ord('a')  # a=0, b=1, c=2, d=3
    row = 8 - int(pos[1])         # 8->0, 7->1, ..., 1->7
    return (row, col)

def load_policy(model_path: str):
    # start model
    policy = MuZeroPolicy(cfg=main_config.policy, enable_field=['eval'])
    state_dict = torch.load(model_path, map_location='cpu')
    policy._model.load_state_dict(state_dict['model'])
    policy_evl = policy.eval_mode
    return policy_evl

def main():
    model_path = sys.argv[2]
    policy = load_policy(model_path)
    
    # init environment
    env = DarkchessEnv(main_config.env)
    # start game
    obs = env.reset()  
    
    print("LightZero MGTP Interface Ready")
    sys.stdout.flush()
    
    while True:
        # get input command
        command = input().strip()
        parts = command.split()
        id = int(command.split()[0])
        print(f"[RECV] {command}", file=sys.stderr)

        if id == 0: # protocol_version
            result = "1.1.0"
            
        elif id == 1:  # name
            result = "LightZero_StochasticMuZero"
             
        elif id == 2:  # version
            result = "1.0.0"
            
        elif id == 5: # quit
            break
        
        elif id == 7:  # reset_board
            obs = env.reset()
            
        elif id == 10:  # move:把對手的棋步傳給模型
            # 格式: "10 move a1 b1"
            from_pos = parts[2]  # "a1"
            to_pos = parts[3]    # "b1"
            
            from_idx = pos_to_coord(from_pos)
            to_idx = pos_to_coord(to_pos)
            
            # 將 (from, to) 轉成 action_id
            action_id = env.action_to_string(from_idx, to_idx)
            obs, _, _, _ = env.step(action_id)
        
        elif id == 11:  # flip:把對手的翻棋傳給模型
            # TODO: 解析對手翻棋並更新環境
            pass
        
        elif id == 12: #genmove:讓AI產生走步
            # model reasoning
            output = policy.forward(obs['observation'], obs['action_mask'])
            action = output['action']
            # TODO: 將 action 轉成 "a1 b1" 格式
            result = env.action_to_string(action)
            obs, _, _, _ = env.step(action)
            
        elif id == 17:  # showboard
            env.render()
            
        # else:
        #     print(f"unknown command")

        print(f"={id} {result}")
        sys.stdout.flush()
            
if __name__ == "__main__":
    main()