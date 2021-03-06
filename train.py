import gym
from ray.tune.logger import pretty_print
from ray.tune.session import report
import gym_hideseek
from gym import logger

import ray
from ray import tune

def _get_args():
    import argparse
    parser = argparse.ArgumentParser()

    train_args = parser.add_argument_group('Training Arguments')
    train_args.add_argument('-N', '--num-workers', default=4, type=int, help='Number of workers')
    train_args.add_argument('-t', '--as-test', action='store_true', help='Run as a test')
    train_args.add_argument('--no-tune', action='store_true')
    train_args.add_argument('-m', '--method', type=str, default='PPO', help='The RLlib-registered algorithm to use')
    train_args.add_argument('--stop-iters', type=int, default=50, help='Number of iterations to train')
    train_args.add_argument('--stop-timesteps', type=int, default=100000, help='Number of timesteps to train')
    train_args.add_argument('--stop-reward', type=float, default=4., help='Reward at which we stop training')
    train_args.add_argument('-C', '--model-config', type=str, required=True, help='The model config file')
    train_args.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    env_args = parser.add_argument_group('Environment Arguments')
    env_args.add_argument('-E', '--env', type=str, required=True, help='Environment')
    env_args.add_argument('--env-config', type=str, required=True, help='The environment config file')
    env_args.add_argument('--render-env', action='store_true')
    env_args.add_argument('--record-env', nargs='?', default=False, const=True)

    args = parser.parse_args()
    return args

def _get_config(args):
    import gym_hideseek.env
    from ray.tune.registry import register_env
    if args.env == 'TD-Hider':
        register_env('TD-Hider-v0', lambda c: gym_hideseek.env.Hider(**c))
        env = 'TD-Hider-v0'
    elif args.env == 'TD-Seeker':
        register_env('TD-Seeker-v0', lambda c: gym_hideseek.env.Seeker(**c))
        env = 'TD-Seeker-v0'
    else:
        logger.warn('main', 'Unknown environment {}', args.env)
        env = args.env
    
    import json
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)
    with open(args.env_config, 'r') as f:
        env_config = json.load(f)
    
    config = {
        'env': env,
        'env_config': env_config,
        'framework': 'torch',
        'num_workers': args.num_workers,
        'model': model_config,
        'lr': args.lr,
        # 'render_env': args.render_env,
        # 'record_env': args.record_env,
        'num_gpus': 1,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    return config, stop

if __name__ == '__main__':
    args = _get_args()
    config, stop = _get_config(args)

    ray.init()

    if args.no_tune:
        if args.method != 'PPO':
            raise ValueError("Only support --method PPO with --no-tune")
        from ray.rllib.agents import ppo
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        trainer = ppo.PPOTrainer(config = ppo_config, env = config['env'])
        for _ in range(args.stop_iters):
            result = trainer.train()
            print(pretty_print(result))
            if result["timesteps_total"] >= args.stop_timesteps or \
                result["episode_reward_mean"] >= args.stop_reward:
                break
    else:
        from ray.tune import CLIReporter
        reporter = CLIReporter(max_progress_rows=10, metric_columns=['episode_reward_mean', 'episodes_this_iter'])
        results = tune.run(args.method, config=config, stop=stop, progress_reporter=reporter)
        if args.as_test:
            from ray.rllib.utils.test_utils import check_learning_achieved
            check_learning_achieved(results, args.stop_reward)
    
    ray.shutdown()
