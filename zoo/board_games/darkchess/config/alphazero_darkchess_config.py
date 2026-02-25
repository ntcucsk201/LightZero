from easydict import EasyDict
from datetime import datetime

# ==============================================================
# begin of the most frequently changed config specified by the user
# 訓練與環境的主要參數，例如環境數、MCTS 模擬次數、batch 大小、棋盤大小、動作空間等
# ==============================================================
use_ture_chance_label_in_chance_encoder = True
collector_env_num = 16    
n_episode = 16
evaluator_env_num = 8
num_simulations = 1000    # MCTS 模擬次數
update_per_collect = 50
reanalyze_ratio = 0
batch_size = 512
max_env_step = int(1e9)
max_train_iter = 2000

board_width = 4
board_height = 8
board_feature_layer = 17
no_eat_flip = 180
long_catch = 3
action_space_size = 352
chance_space_size = 14
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

darkchess_alphazero_config = dict(
    # TODO: 依所需資訊更改檔名
    exp_name=
    f'data_alphazero/darkchess_alphazero_ns{num_simulations}_upc{update_per_collect}_trainitr{max_train_iter}_bs{batch_size}_{timestamp}',
    env=dict(                          #設定環境參數，包括觀察空間、棋盤大小、對戰模式等
        env_id='darkchess',
        obs_shape=(17, 8, 4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        board_width=board_width,
        board_height=board_height,
        board_feature_layer=board_feature_layer,
        battle_mode='self_play_mode',
        battle_mode_in_simulation_env='self_play_mode',  # only used in AlphaZero
        long_catch=long_catch,
        no_eat_flip=no_eat_flip,
        agent_vs_human=False,
    ),
    policy=dict(                            # 設定訓練策略、模型結構、學習率、MCTS 參數、buffer 大小等
        # TODO:
        model=dict(
            observation_shape=(17, 8, 4),
            action_space_size=action_space_size,
        ),
        learn=dict(
            learner=dict(
                train_iterations=max_train_iter,
                hook=dict(
                    load_ckpt_before_run='',      # 開始訓練前不載入 checkpoint
                    log_show_after_iter=100,      # 每訓練 100 iter 印一次leaner訓練狀態
                    save_ckpt_after_iter=1000,   # 每訓練 1000 iter 存一次 checkpoint
                    save_ckpt_after_run=True,     # 訓練結束（自然結束/手動停止）時 再存一次 checkpoint
                ),
            ),
            resume_training=False,  # 從頭訓練
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        # use_ture_chance_label_in_chance_encoder=use_ture_chance_label_in_chance_encoder,
        env_type='board_games',
        mcts_ctree=True,
        action_type='varied_action_space',
        cuda=True,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        entropy_weight=0.001,
        simulation_env_id='darkchess',
        simulation_env_config_type='self_play',

        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=int(board_width * board_height / 2),  # for battle_mode='play_with_bot_mode'
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=1,
        weight_decay=1e-4,
        game_segment_length=200,  # TODO:

    ),
)
darkchess_alphazero_config = EasyDict(darkchess_alphazero_config)
main_config = darkchess_alphazero_config

darkchess_alphazero_create_config = dict(         # 建立環境與 policy 的設定，指定要用哪個環境、哪個 policy 類別
    env=dict(
        type='darkchess',
        import_names=['zoo.board_games.darkchess.envs.darkchess_alphazero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
)
darkchess_alphazero_create_config = EasyDict(darkchess_alphazero_create_config)
create_config = darkchess_alphazero_create_config

if __name__ == "__main__":
    # Define a list of seeds for multiple runs
    seeds = [0]  # You can add more seed values here
    # for seed in seeds:
        # Update exp_name to include the current seed
        # main_config.exp_name = f'data_stochastic_mz/darkchess_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed{seed}'
    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
 