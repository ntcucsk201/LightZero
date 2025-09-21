from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
use_ture_chance_label_in_chance_encoder = True
collector_env_num = 16
n_episode = 16
evaluator_env_num = 8
num_simulations = 1000
update_per_collect = 50
reanalyze_ratio = 0
batch_size = 512
max_env_step = 300000                   # int(1e6)
# max_train_iter = int(1e6)

board_width = 4
board_height = 8
board_feature_layer = 16
no_eat_flip = 180
long_catch = 3
action_space_size = 352
chance_space_size = 14
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

darkchess_stochastic_muzero_config = dict(
    # TODO: 依所需資訊更改檔名
    exp_name=
    f'data_stochastic_mz/darkchess_stochastic_muzero_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_bs{batch_size}',
    env=dict(
        env_id='darkchess',
        obs_shape=(16, 8, 4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        board_width=board_width,
        board_height=board_height,
        board_feature_layer=board_feature_layer,
        battle_mode='self_play_mode',
        long_catch=long_catch,
        no_eat_flip=no_eat_flip,
        agent_vs_human=False,
    ),
    policy=dict(
        # TODO:
        model=dict(
            observation_shape=(16, 8, 4),
            action_space_size=action_space_size,
            chance_space_size=chance_space_size,
            image_channel=16,
            # NOTE: whether to use the self_supervised_learning_loss. default is False
            self_supervised_learning_loss=True,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        # use_ture_chance_label_in_chance_encoder=use_ture_chance_label_in_chance_encoder,
        env_type='board_games',
        action_type='varied_action_space',
        cuda=True,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.005,
        ssl_loss_weight=2,  # NOTE: default is 0.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,

        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=int(board_width * board_height / 2),  # for battle_mode='play_with_bot_mode'
        # NOTE：In board_games, we set discount_factor=1.
        discount_factor=1,
        weight_decay=1e-4,
        game_segment_length=200,
        
        # 要接續哪個checkpoint繼續訓練
        learn=dict(
        learner=dict(
            hook=dict(
                load_ckpt_before_run="/home/ntcucsk201/DarkChess/LightZero/zoo/board_games/darkchess/data_stochastic_mz/darkchess_stochastic_muzero_ns1000_upc50_rer0_bs512_250918_141430/ckpt/iteration_0.pth.tar",  # 指定要讀的 checkpoint
            ),
        ),
    ),
    ),
)
darkchess_stochastic_muzero_config = EasyDict(darkchess_stochastic_muzero_config)
main_config = darkchess_stochastic_muzero_config

darkchess_stochastic_muzero_create_config = dict(
    env=dict(
        type='darkchess',
        import_names=['zoo.board_games.darkchess.envs.darkchess_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero',
        import_names=['lzero.policy.stochastic_muzero'],
    ),
)
darkchess_stochastic_muzero_create_config = EasyDict(darkchess_stochastic_muzero_create_config)
create_config = darkchess_stochastic_muzero_create_config

if __name__ == "__main__":
    # Define a list of seeds for multiple runs
    seeds = [0]  # You can add more seed values here
    # for seed in seeds:
        # Update exp_name to include the current seed
        # main_config.exp_name = f'data_stochastic_mz/darkchess_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed{seed}'
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step, model_path=main_config.policy.model_path)
