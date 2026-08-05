"""Built-in Fleet Scheduling Policies.  These classes can be extended for
future research."""
from enum import Enum
from typing import Dict

import argparse
import datetime
import json
import logging
import pickle
import random

import coloredlogs
import gymnasium as gym
import numpy
import yaml

from scipy import stats

from simulator.job import *
from simulator.vehicle import *
from simulator.charger import *
from simulator.demand import *
from simulator.simulator import *

import stable_baselines3
import torch


class SchedulePolicy:
    """Abstract Policy Class."""

    def __init__(self) -> None:
        pass

    def schedule(self, observation: numpy.array, info: Dict) -> numpy.array:
        """Compute a schedule given observations and info."""
        raise NotImplemented


class EightyTwentyPolicy(SchedulePolicy):
    """Charge vehicles at maximum available rate to 80% SoC, vehicles service
    demand until SoC drops below 20%, at which point they return to the
    nearest charger.
    """

    def __init__(self):
        super().__init__()

    def schedule(self, observation: numpy.array, info: Dict) -> numpy.array:
        action = numpy.zeros((50, 2))
        for v in range(len(info["fleet"])):
            if observation[v, 1] < 0.2:
                action[v, 0] = 1
                action[v, 1] = 72.1
        return action


class DnnPolicy(SchedulePolicy):
    """A DNN takes the SoC and SoH of each each vehicle and returns whether
    the vehicle should be chargning and if so how fast.
    """
    
    def __init__(self, weights: str) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dnn = torch.load(weights, weights_only=False).eval().to(self.device)

    def schedule(self, observation, info):
        with torch.no_grad():
            x = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            action = self.dnn(x)[0].squeeze().cpu().detach().numpy()
            action[:, 1] = action[:, 1] * 50.0 # scale from normalized [-1, 1] to max port power 50kW
            return action


class DataLogger:
    """Get data for plots."""

    def __init__(self, logfile):
        self.csvfile = open(logfile, "w")
        self.csvfile.write("profit,total_power,completed,")
        self.csvfile.write(",".join([f"soh{i}" for i in range(50)]))
        self.csvfile.write(",")
        self.csvfile.write(",".join([f"status{i}" for i in range(50)]))
        self.csvfile.write("\n")
        self.p_old = [72.1] * 50
        self.retired = [0] * 50

    def write(self, info):
        total_power = info.get("total_grid_power", 0.0)
        p_curr = []
        soh_curr = []
        state = []
        for v in range(50):
            p_curr.append(info["fleet"][v]["battery"]["soc"] * 72.1)
            if info["fleet"][v]["battery"]["actual_capacity"] / 72.1 <= 0.8:
                self.retired[v] = 1
            soh_curr.append(
                info["fleet"][v]["battery"]["actual_capacity"]
                / info["fleet"][v]["battery"]["initial_capacity"]
            )
            state.append(1 if info["fleet"][v]["status"] == "RECOVERY" else 0)
        self.p_old = p_curr

        profit = 0
        for j in info["inprogress"]:
            if self.retired[j["vehicle"]] < 1:
                profit += j["fare"]

        entry = f"{profit},{total_power},"
        for i in range(50):
            entry += f"{soh_curr[i]},"
        entry += ",".join([f"{state[i]}" for i in range(50)])
        self.csvfile.write(entry + "\n")

    def close(self):
        self.csvfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate vehicle fleet")
    parser.add_argument(
        "-c", "--config", help="Path to configuration file for a simulation"
    )
    parser.add_argument("-o", "--output", help="Path to state output log")
    parser.add_argument("-p", "--policy", help="EIGHTYTWENTY or DNN")
    parser.add_argument("-w", "--weights", help="Path to policy weights for DNN")
    args = parser.parse_args()

    config = {}
    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp.read())

    datalogger = DataLogger(args.output)

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
