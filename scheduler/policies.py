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

import os
import sys
from pathlib import Path

import sys
from pathlib import Path
# Add the project root to Python's path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
            # if observation[v, 1] < 0.2:
            # Access the SoC directly from the unflattened info dictionary
            if info["fleet"][v]["battery"]["soc"] < 0.2:
                action[v, 0] = 1
                action[v, 1] = 72.1
        return action


# class DnnPolicy(SchedulePolicy):
#     """A DNN takes the SoC and SoH of each each vehicle and returns whether
#     the vehicle should be chargning and if so how fast.
#     """
    
#     def __init__(self, weights: str) -> None:
#         super().__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.dnn = torch.load(weights, weights_only=False).eval().to(self.device)

#     def schedule(self, observation, info):
#         with torch.no_grad():
#             x = torch.from_numpy(observation).unsqueeze(0).to(self.device)
#             action = self.dnn(x)[0].squeeze().cpu().detach().numpy()
#             action[:, 1] = action[:, 1] * 50.0 # scale from normalized [-1, 1] to max port power 50kW
#             return action
        
from stable_baselines3 import PPO

class DnnPolicy(SchedulePolicy):
    """Loads a trained SB3 PPO agent and returns determinisitic actions."""
    
    def __init__(self, weights: str) -> None:
        super().__init__()
        # Load standard SB3 Zip file, not raw Torch
        self.model = PPO.load(weights)

    def schedule(self, observation, info):
        # model.predict automatically handles tensor conversions and returns bounded actions
        action, _ = self.model.predict(observation, deterministic=True)
        return action

class DataLogger:
    """Exhaustive Data Logger for full visualization freedom."""

    def __init__(self, logfile):
        self.csvfile = open(logfile, "w")
        
        headers = [
            "Time_Step", "Timestamp", "Vehicle_ID", "Status", "SoC", "SoH",
            "Location_Zone", "Destination_Zone", "Distance_Remaining_km", "Time_Remaining_sec",
            "Net_Building_Load_kW", "Total_Grid_Power_kW", "Grid_Price",
            "V2B_Revenue_Global", "Completed_Rides_Global", "Skipped_Rides_Global",
            "Active_Jobs", "Failed_Jobs"
        ]
        self.csvfile.write(",".join(headers) + "\n")
        self.step_count = 0

    def write(self, info, timestamp):
        # Extract global environment metrics
        net_building_load = info.get("net_building_load", 0.0)
        total_grid_power = info.get("total_grid_power", 0.0)
        grid_price = info.get("current_price", 0.0)
        v2b_rev = info.get("V2B_Revenue", 0.0)
        completed = info.get("step_completed", 0)
        skipped = info.get("rejected", 0) # Mapped from the updated info dict
        active = len(info.get("inprogress", []))
        failed = info.get("failed", 0)

        # Write a row for EVERY vehicle
        for v_idx, v in enumerate(info.get("fleet", [])):
            soc = v["battery"]["soc"]
            soh = v["battery"]["actual_capacity"] / v["battery"]["initial_capacity"]
            status = str(v["status"]).split(".")[-1]
            loc = v.get("location", "Unknown")
            dest = v.get("destination", "Unknown")
            dist = v.get("distance_remaining", 0.0)
            t_rem = v.get("time_remaining", 0.0)

            row = [
                str(self.step_count), str(timestamp), str(v_idx), status,
                f"{soc:.4f}", f"{soh:.4f}", str(loc), str(dest), f"{dist:.2f}", f"{t_rem:.1f}",
                f"{net_building_load:.2f}", f"{total_grid_power:.2f}", f"{grid_price:.4f}",
                f"{v2b_rev:.4f}", str(completed), str(skipped), str(active), str(failed)
            ]
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
