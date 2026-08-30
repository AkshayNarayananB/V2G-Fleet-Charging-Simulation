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
        
        # Shifted to Long-Format headers to match the pandas pipeline
        headers = [
            "Time_Step", "Timestamp", "Vehicle_ID", "State", "SoC", "SoH", 
            "Power_kW", "Building_Load_kW", "Total_Grid_Power", "Grid_Price", 
            "Completed_Rides_Global", "Skipped_Rides_Global", "Taxi_Profit_Global", "V2B_Revenue_Global"
        ]
        self.csvfile.write(",".join(headers) + "\n")
        self.retired = [0] * 50
        self.step_count = 0

    def write(self, info, timestamp):
        # Extract global environment metrics
        net_building_load = info.get("net_building_load", 0.0)
        total_grid_power = info.get("total_grid_power", 0.0)
        
        # Ensure skipped rides is passed from your gym env into the info dict
        completed_rides = info.get("step_completed", info.get("completed_rides", 0))
        skipped_rides = info.get("skipped_rides", 0) 
        
        v2b_revenue = info.get("V2B_Revenue", 0.0)
        current_price = info.get("current_price", 0.0)

        # Calculate live taxi profit
        profit = sum([j["fare"] for j in info.get("inprogress", []) if self.retired[j["vehicle"]] < 1])

        # Write a row for EVERY vehicle to enable the state matrix and degradation tracking
        for v in range(50):
            battery = info["fleet"][v]["battery"]
            actual_cap = battery["actual_capacity"]
            initial_cap = battery["initial_capacity"]
            
            # Mark vehicle as retired if capacity drops below 80%
            if actual_cap / initial_cap <= 0.8:
                self.retired[v] = 1
                
            soc = battery["soc"]
            soh = actual_cap / initial_cap
            
            # Extract the pure string name of the Enum (e.g., 'IDLE', 'DRIVING')
            state = str(info["fleet"][v]["status"]).split('.')[-1]
            
            # Fetch the applied power. Change the key below if your env uses a different name (e.g., 'charge_rate')
            # Negative values = Discharging (V2G), Positive = Charging
            power_kw = info["fleet"][v].get("power_kw", 0.0) 

            row = [
                str(self.step_count),
                str(timestamp),
                str(v),
                state,
                f"{soc:.4f}",
                f"{soh:.4f}",
                f"{power_kw:.4f}",
                f"{net_building_load:.2f}",
                f"{total_grid_power:.2f}",
                f"{current_price:.4f}",
                str(completed_rides),
                str(skipped_rides),
                f"{profit:.2f}",
                f"{v2b_revenue:.4f}"
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
