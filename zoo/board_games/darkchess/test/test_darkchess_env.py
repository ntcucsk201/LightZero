import pytest
from easydict import EasyDict

from zoo.board_games.darkchess.envs.darkchess_env import DarkchessEnv


@pytest.mark.envtest
class TestDarkchessEnv:

    def test_self_play_mode(self):
        cfg = DarkchessEnv.default_config()
        env = DarkchessEnv(cfg)
        obs = env.reset()
        print(len(env.all_actions))
        print('init board state: ')
        while True:
            action = env.random_action()
            # action = env.human_to_action()
            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            if done:
                if reward > 0:
                    print('player 1 (human player) win')
                else:
                    print('draw')
                break
            # action = env.bot_action()
            action = env.random_action()
            # action = env.human_to_action()
            print('player 2 (computer player): ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            if done:
                if reward > 0:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break
        print(env.player_color)
        print(len(env.action_history))
        print(env.chess_count)
        print(env.flipped_chess_count)


test = TestDarkchessEnv()
test.test_self_play_mode()
