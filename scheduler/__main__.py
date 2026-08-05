"""Built-in Fleet Scheduling Policies.  These classes can be extended for
future research."""

import argparse
import datetime
import json
import logging
import pickle
import random

import coloredlogs
import gymnasium as gym
import numpy
import torch
import yaml

from scipy import stats
from stable_baselines3 import PPO

from simulator.job import *
from simulator.vehicle import *
from simulator.charger import *
from simulator.demand import *
from simulator.simulator import *

from scheduler.policies import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate vehicle fleet")
    parser.add_argument(
        "-c", "--config", help="Path to configuration file for a simulation"
    )
    parser.add_argument("-a", "--action", help="TRAIN or EVAL")
    parser.add_argument("-o", "--output", help="Path to state output log")
    parser.add_argument("-p", "--policy", help="EIGHTYTWENTY or DNN")
    parser.add_argument("-w", "--weights", help="Path to policy weights for DNN")
    parser.add_argument("--epochs", type=int, help="Number of epochs (training)")
    args = parser.parse_args()

    config = {}
    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp.read())

    datalogger = DataLogger(args.output)

    if args.action.lower() == 'train':
        env = TaxiFleetSimulator(config)
        env.reset()
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=args.epochs)
        torch.save(model.policy, "ppo_policy.pt")

    elif args.action.lower() == 'eval':
        policy = None
        if args.policy.lower() == "eightytwenty":
            policy = EightyTwentyPolicy()
        elif args.policy.lower() == "dnn":
            policy = DnnPolicy(args.weights)
        else:
            raise Exception("Choose a supported policy!")

        environment = TaxiFleetSimulator(config)
        observation, info = environment.reset()
        done = False

        while not done:
            datalogger.write(info)
            action = policy.schedule(observation, info)
            observation, reward, done, _, info = environment.step(action)

        datalogger.close()

    else:
        print('Must choose TRAIN or EVAL')
