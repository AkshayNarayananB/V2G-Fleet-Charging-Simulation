import argparse
import yaml
import gymnasium as gym

# Note: Adjust these imports if your directory structure differs slightly.
from simulator.simulator import TaxiFleetSimulator
from scheduler.policies import EightyTwentyPolicy, DnnPolicy, DataLogger

# Assuming Stable Baselines 3 is used for the DNN RL training based on the standard Gym architecture
from stable_baselines3 import PPO 

def main():
    parser = argparse.ArgumentParser(description="Taxi Fleet Scheduler")
    parser.add_argument('-a', '--action', required=True, choices=['TRAIN', 'EVAL', 'train', 'eval'], help="Action to perform")
    parser.add_argument('-c', '--config', required=True, help="Path to config.yaml")
    parser.add_argument('-w', '--weights', help="Path to weights (output for TRAIN, input for EVAL DNN)")
    parser.add_argument('--epochs', type=int, default=10000, help="Number of timesteps/epochs to train on")
    parser.add_argument('-p', '--policy', choices=['EIGHTYTWENTY', 'DNN', 'eightytwenty', 'dnn'], help="Policy type (EVAL only)")
    parser.add_argument('-o', '--output', default='results.csv', help="Path to csv log output (EVAL only)")

    args = parser.parse_args()

    # Load the YAML configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize the customized Gym Environment
    environment = TaxiFleetSimulator(config)

    if args.action.upper() == 'TRAIN':
        print(f"Starting PPO training for {args.epochs} timesteps...")
        # Initialize the PPO agent
        model = PPO("MlpPolicy", environment, verbose=1)
        
        # Train the model
        model.learn(total_timesteps=args.epochs)
        
        # Save the resulting weights
        if args.weights:
            model.save(args.weights)
            print(f"Training complete. Weights saved to {args.weights}")
        else:
            print("Training complete. No weights output path provided (-w).")

    elif args.action.upper() == 'EVAL':
        if not args.output:
            raise ValueError("Evaluation requires an output CSV path (-o).")
            
        datalogger = DataLogger(args.output)
        
        # Load the selected policy
        if args.policy.upper() == "EIGHTYTWENTY":
            policy = EightyTwentyPolicy()
        elif args.policy.upper() == "DNN":
            if not args.weights:
                raise ValueError("DNN policy requires a weights file (-w).")
            policy = DnnPolicy(args.weights)
        else:
            raise ValueError("Choose a supported policy! (-p EIGHTYTWENTY or DNN)")

        print(f"Evaluating {args.policy.upper()} policy...")
        
        # Reset environment for evaluation
        observation, info = environment.reset()
        done = False
        truncated = False

        # Evaluation Loop
        while not (done or truncated):
            # 1. Log current state BEFORE taking action, passing the internal environment time
            datalogger.write(info, environment.t)
            
            # 2. Policy determines action based on observation
            action = policy.schedule(observation, info)
            
            # 3. Step the environment forward
            observation, reward, done, truncated, info = environment.step(action)

        # Log the final timestep state
        datalogger.write(info, environment.t)
        datalogger.close()
        
        print(f"Evaluation complete. Results saved to {args.output}")

if __name__ == '__main__':
    main()
