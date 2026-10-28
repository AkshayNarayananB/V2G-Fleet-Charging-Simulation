# import argparse
# import yaml
# import gymnasium as gym

# from simulator.simulator import TaxiFleetSimulator
# from scheduler.policies import EightyTwentyPolicy, DnnPolicy, DataLogger
# from stable_baselines3 import PPO 
# from stable_baselines3.common.callbacks import CheckpointCallback

# def main():
#     parser = argparse.ArgumentParser(description="Taxi Fleet Scheduler")
#     parser.add_argument('-a', '--action', required=True, choices=['TRAIN', 'EVAL', 'train', 'eval'])
#     parser.add_argument('-c', '--config', required=True)
#     parser.add_argument('-w', '--weights')
#     parser.add_argument('--epochs', type=int, default=100000)
#     parser.add_argument('-p', '--policy', choices=['EIGHTYTWENTY', 'DNN', 'eightytwenty', 'dnn'])
#     parser.add_argument('-o', '--output', default='results.csv')

#     args = parser.parse_args()

#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)

#     environment = TaxiFleetSimulator(config)

#     if args.action.upper() == 'TRAIN':
#         print(f"Starting PPO training for {args.epochs} timesteps...")
        
#         # Save a checkpoint every 10,000 steps
#         checkpoint_callback = CheckpointCallback(
#             save_freq=10000, 
#             save_path='./checkpoints/',
#             name_prefix='ppo_nyc_model'
#         )
        
#         model = PPO("MlpPolicy", environment, verbose=1)
#         # Pass the callback to the learning loop
#         model.learn(total_timesteps=args.epochs, callback=checkpoint_callback)
        
#         if args.weights:
#             model.save(args.weights)
#             print(f"Training complete. Final weights saved to {args.weights}")

#     elif args.action.upper() == 'EVAL':
#         if not args.output:
#             raise ValueError("Evaluation requires an output CSV path (-o).")
            
#         datalogger = DataLogger(args.output)
        
#         if args.policy.upper() == "EIGHTYTWENTY":
#             policy = EightyTwentyPolicy()
#         elif args.policy.upper() == "DNN":
#             if not args.weights:
#                 raise ValueError("DNN policy requires a weights file (-w).")
#             policy = DnnPolicy(args.weights)
#         else:
#             raise ValueError("Choose a supported policy! (-p EIGHTYTWENTY or DNN)")

#         print(f"Evaluating {args.policy.upper()} policy...")
        
#         observation, info = environment.reset()
#         done = False
#         truncated = False

#         while not (done or truncated):
#             datalogger.write(info, environment.t)
#             action = policy.schedule(observation, info)
#             observation, reward, done, truncated, info = environment.step(action)

#         datalogger.write(info, environment.t)
#         datalogger.close()
#         print(f"Evaluation complete. Results saved to {args.output}")

# if __name__ == '__main__':
#     main()

# Abive is the earlier stable baseline code
# Below is thwe PPO - Langrarian code

import argparse
import yaml
import numpy as np
import gymnasium as gym

from simulator.simulator import TaxiFleetSimulator
from scheduler.policies import EightyTwentyPolicy, DnnPolicy, DataLogger
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class PPOLagrangianCallback(BaseCallback):
    """Dynamically updates the penalty multiplier based on constraint violations."""
    def __init__(self, cost_limit=0.0, lr_lambda=0.01, verbose=1):
        super().__init__(verbose)
        self.cost_limit = cost_limit
        self.lr_lambda = lr_lambda
        self.episode_costs = []
        self.current_cost = 0.0

    def _on_step(self) -> bool:
        # Fetch the peak violation from the environment's info dictionary
        violation = self.locals['infos'][0].get('peak_violation', 0.0)
        self.current_cost += violation
        
        if self.locals['dones'][0]:
            self.episode_costs.append(self.current_cost)
            self.current_cost = 0.0
            
        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_costs) > 0:
            mean_cost = np.mean(self.episode_costs)
            self.episode_costs = []
            
            # Access the base Gym environment inside the DummyVecEnv wrapper
            env = self.training_env.envs[0].unwrapped
            
            # Dual Gradient Ascent: Update the Lagrangian multiplier
            new_lambda = env.lagrangian_multiplier + self.lr_lambda * (mean_cost - self.cost_limit)
            env.lagrangian_multiplier = max(0.0, new_lambda)
            
            if self.verbose > 0:
                print(f"\n[CMDP] Mean Peak Violation: {mean_cost:.2f} kW | New Penalty Multiplier (Lambda): {env.lagrangian_multiplier:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Taxi Fleet Scheduler")
    parser.add_argument('-a', '--action', required=True, choices=['TRAIN', 'EVAL', 'train', 'eval'])
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-w', '--weights')
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('-p', '--policy', choices=['EIGHTYTWENTY', 'DNN', 'eightytwenty', 'dnn'])
    parser.add_argument('-o', '--output', default='results.csv')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    environment = TaxiFleetSimulator(config)

    if args.action.upper() == 'TRAIN':
        print(f"Starting CMDP PPO-Lagrangian training for {args.epochs} timesteps...")
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path='./checkpoints/',
            name_prefix='ppo_cmdp_model'
        )
        lagrangian_callback = PPOLagrangianCallback(cost_limit=0.0, lr_lambda=0.01)
        
        model = PPO("MlpPolicy", environment, verbose=1)
        # Pass BOTH callbacks to the learning loop
        model.learn(total_timesteps=args.epochs, callback=[checkpoint_callback, lagrangian_callback])
        
        if args.weights:
            model.save(args.weights)
            print(f"Training complete. Weights saved to {args.weights}")

    elif args.action.upper() == 'EVAL':
        if not args.output:
            raise ValueError("Evaluation requires an output CSV path (-o).")
            
        datalogger = DataLogger(args.output)
        
        if args.policy.upper() == "EIGHTYTWENTY":
            policy = EightyTwentyPolicy()
            use_sb3_direct = False
        elif args.policy.upper() == "DNN":
            model = PPO.load(args.weights)
            use_sb3_direct = True

        print(f"Evaluating {args.policy.upper()} policy...")
        
        observation, info = environment.reset()
        done = False
        truncated = False

        while not (done or truncated):
            datalogger.write(info, environment.t)
            if use_sb3_direct:
                action, _ = model.predict(observation, deterministic=True)
            else:
                action = policy.schedule(observation, info)
            observation, reward, done, truncated, info = environment.step(action)

        datalogger.write(info, environment.t)
        datalogger.close()
        print(f"Evaluation complete. Results saved to {args.output}")

if __name__ == '__main__':
    main()