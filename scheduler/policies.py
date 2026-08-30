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
    """Get data for plots and analysis."""

    def __init__(self, logfile):
        self.csvfile = open(logfile, "w")
        
        # Define core telemetry headers
        headers = [
            "step", "timestamp", "net_building_load", "total_grid_power",
            "step_completed", "v2b_revenue", "current_price", "taxi_profit"
        ]
        # Append per-vehicle headers
        headers.extend([f"soh_{i}" for i in range(50)])
        headers.extend([f"soc_{i}" for i in range(50)])
        headers.extend([f"status_{i}" for i in range(50)])
        
        self.csvfile.write(",".join(headers) + "\n")
        self.retired = [0] * 50
        self.step_count = 0

    def write(self, info, timestamp):
        # Extract environment metrics
        net_building_load = info.get("net_building_load", 0.0)
        total_grid_power = info.get("total_grid_power", 0.0)
        step_completed = info.get("step_completed", 0)
        v2b_revenue = info.get("V2B_Revenue", 0.0)
        current_price = info.get("current_price", 0.0)

        soh_curr = []
        soc_curr = []
        state = []
        
        for v in range(50):
            soc_curr.append(info["fleet"][v]["battery"]["soc"])
            # Mark vehicle as retired if capacity drops below 80%
            if info["fleet"][v]["battery"]["actual_capacity"] / info["fleet"][v]["battery"]["initial_capacity"] <= 0.8:
                self.retired[v] = 1
            soh_curr.append(
                info["fleet"][v]["battery"]["actual_capacity"] / info["fleet"][v]["battery"]["initial_capacity"]
            )
            # Extract enum name (e.g., 'IDLE')
            state.append(str(info["fleet"][v]["status"]).split('.')[-1])

        # Calculate live taxi profit
        profit = sum([j["fare"] for j in info["inprogress"] if self.retired[j["vehicle"]] < 1])

        # Construct row array
        row = [
            str(self.step_count), 
            str(timestamp), 
            f"{net_building_load:.2f}", 
            f"{total_grid_power:.2f}",
            str(step_completed), 
            f"{v2b_revenue:.4f}", 
            f"{current_price:.4f}", 
            f"{profit:.2f}"
        ]
        
        row.extend([f"{soh:.4f}" for soh in soh_curr])
        row.extend([f"{soc:.4f}" for soc in soc_curr])
        row.extend(state)
        
        self.csvfile.write(",".join(row) + "\n")
        self.step_count += 1

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
