import sys
import torch
from lzero.policy.muzero import MuZeroPolicy
from zoo.board_games.darkchess.config.muzero_darkchess_config import main_config, create_config
from lzero.envs.get_wrapped_env import get_wrappered_env

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
    env_fn = get_wrappered_env(create_config, env_id="darkchess")
    env = env_fn()  
    # start game
    obs = env.reset()  
    
    print("LightZero MGTP Interface Ready")
    sys.stdout.flush()
    
    while True:
        # get input command
        command = input().strip()
        id = int(command.split()[0])
        print(f"[RECV] {command}", file=sys.stderr)

        if id ==5: # quit
            break
        
        elif command.startswith("genmove"):
            # model reasoning
            output = policy.eval_mode.forward(obs)
            next_obs, rew, done, info = env.step(output.action)
            print(f"= {output.action}")
            sys.stdout.flush()
            
        elif command.startswith("showboard"):
            print("=")
            print(env.render('human'))
            sys.stdout.flush()

        else:
            print("? unknown command")
            sys.stdout.flush()
            
if __name__ == "__main__":
    main()