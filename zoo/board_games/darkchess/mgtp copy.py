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

def mgtp_to_action_id(from_pos: str, to_pos: str) -> int:
    """
    將 MGTP 座標轉成 action_id
    例如: "a1", "b1" -> action_id
    """
    from_coord = pos_to_coord(from_pos)  # "a1" -> (7, 0)
    to_coord = pos_to_coord(to_pos)      # "b1" -> (7, 1)
    action = (from_coord, to_coord)      # ((7, 0), (7, 1))
    return DarkchessEnv.all_actions.index(action)  # 找出 action_id

# def load_policy(model_path: str):
#     # 1) 一次初始化 learn + eval(初始化一個推論用的結構)，確保 inverse_scalar_transform_handle 會被建立
#     policy = MuZeroPolicy(cfg=main_config.policy, enable_field=['eval'])

#     ckpt = torch.load(model_path, map_location='cpu')

#     # 2) 把權重載到「底層 model」
#     policy._model.load_state_dict(ckpt['model'], strict=True)

#     # 3) 如果框架裡 eval 用的是 _eval_model，最好也同步
#     if hasattr(policy, '_eval_model') and policy._eval_model is not None:
#         policy._eval_model.load_state_dict(ckpt['model'], strict=True)
#         policy._eval_model.eval()

#     policy._model.eval()

#     # 4) CUDA
#     if torch.cuda.is_available():
#         policy._cfg.device = 'cuda'
#         if hasattr(policy, '_eval_model') and policy._eval_model is not None:
#             policy._eval_model.cuda()
#         policy._model.cuda()

#     return policy

def main():
    model_path = sys.argv[2]
    # policy = load_policy(model_path)
    policy = MuZeroPolicy(cfg=main_config.policy, enable_field=['eval'])
    policy_eval = policy.eval_mode
    
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
            
            # 轉成 action_id
            action_id = mgtp_to_action_id(from_pos, to_pos)
            obs, _, _, _ = env.step(action_id) # 執行動作，更新盤面
        
        elif id == 11:  # flip:把對手的翻棋傳給模型
            # 格式: "11 flip a1 K"
            pos = parts[2]       # "a1"
            # piece = parts[3]   # "K" (棋子類型，環境會自動處理)
            
            # 翻棋: from == to
            action_id = mgtp_to_action_id(pos, pos)
            obs, _, _, _ = env.step(action_id)
        
        elif id == 12: #genmove: 讓AI產生走步
            obs = env.ready_obs
            inference_output = policy_eval.forward(obs)
            next_obs, rew, done, info = env.step(inference_output.action)
            
            # model reasoning
            # obs_tensor = torch.from_numpy(obs['observation']).unsqueeze(0).float() # 依照盤面產生動作
            # if torch.cuda.is_available():
            #     obs_tensor = obs_tensor.cuda()
        
            # action_mask_list = [obs['action_mask'].tolist()]
            # to_play_list = [obs.get('to_play', -1)]
            
            # action = policy.eval_mode.forward(
            #     obs_tensor, action_mask_list, to_play_list
            # )
            
            # action = action[0]["action"]
            
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
