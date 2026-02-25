import subprocess
import sys
import torch
import numpy as np
import importlib.util

from lzero.policy.muzero import MuZeroPolicy
# from zoo.board_games.darkchess.config.muzero_darkchess_config import main_config
from zoo.board_games.darkchess.envs.darkchess_env import DarkchessEnv
from ding.torch_utils import to_tensor

def main():

    model_path = sys.argv[2]
    config_path = sys.argv[4]

    # 動態加載配置文件
    try:
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        main_config = config_module.main_config
        sys.stderr.write(f"[DEBUG] Config loaded successfully\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to load config: {str(e)}\n")
        sys.stderr.flush()
        sys.exit(1)

    # 設定 config 中參數
    main_config.env.battle_mode = 'play_with_bot_mode'
    main_config.env.render_mode = 'state_realtime_mode'
    main_config.env.agent_vs_human = True
    
    # 載入 Policy (確保 cfg 與訓練時一致)
    policy = MuZeroPolicy(cfg=main_config.policy, enable_field=['eval'])
    # 載入模型並檢查是否有可用的 CUDA 將模型移到相同設備
    checkpoint = torch.load(model_path, map_location='cpu')
    policy.eval_mode.load_state_dict(checkpoint)
    # policy_eval = policy.eval_mode
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy._model.to(device)
    sys.stderr.write(f"[DEBUG] Model moved to device: {device}\n")
    sys.stderr.flush()

    # 定義 Bot 推理函數
    def bot_policy_fn(obs: dict) -> int:
        """
        Bot 根據觀察產生動作 ID
        obs: 包含 'observation' 和 'action_mask' 的字典
        返回: action_id (int)
        """
        observation_tensor = to_tensor(obs['observation']).unsqueeze(0).to(device)
        
        with torch.no_grad():
            inference_output = policy._forward_eval(
                data=observation_tensor,
                action_mask=[obs['action_mask']],
                to_play=[-1]
            )
        
        action = inference_output[0]['action']
        if isinstance(action, torch.Tensor):
            action = action.item()
        
        return action

    env = DarkchessEnv(main_config.env)
    env.bot_policy_fn = bot_policy_fn

    obs = env.reset()

    print("LightZero MGTP Interface Ready")
    sys.stdout.flush()

    while True:
        try:
            command = input().lstrip()      # 讀使用者輸入的一個MGTP指令
            parts = command.split()
            id = parts[0]
            cmd = parts[1] 
            arg = parts[2:]
            result = ""

            if cmd == "protocol_version":      # 0
                result = "1.1.0"
            elif cmd == "name":                # 1
                result = "LightZero_DarkChess"
            elif cmd == "version":             # 2
                result = "1.0.0"
            elif cmd == "quit":                # 5
                print(f"={id}")
                break
            elif cmd == "reset_board":         # 7
                obs = env.reset()      
            elif cmd == "move":                # 10
                # 轉成 action_id 並執行
                from_pos, to_pos = arg[0], arg[1]
                action_id = mgtp_to_action_id(from_pos, to_pos)
                obs, _, _, _ = env.step(action_id)
            elif cmd == "flip":                # 11
                pos = arg[0]
                flip_chess = arg[1]
                action_id = mgtp_to_action_id(pos, pos)
                obs, _, _, _ = env.step(action_id, flip_chess)
            elif cmd == "genmove":             # 12
                # LightZero 模型根據當前 obs 產生走步
                action_id = bot_policy_fn(obs)
                result = env.action_to_string(action_id)
                
                # 執行模型產生的動作，更新環境
                # obs, _, _, _ = env.step(action_id)
            
            # 標準回應格式
            print(f"={id} {result}\n")
            sys.stdout.flush()

        except Exception as e:
            # print(f"Error: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            
def mgtp_to_action_id(from_pos, to_pos):
    """
    對接 DarkchessEnv.all_actions
    """
    def decode(p):
        col = ord(p[0]) - ord('a')
        row = 8 - int(p[1])
        return (row, col)
    
    act = (decode(from_pos), decode(to_pos))
    try:
        return DarkchessEnv.all_actions.index(act)
    except ValueError:
        return -1 # 非法動作處理
    
    
if __name__ == "__main__":
    main()