"""Taxi fleet simulator."""
from typing import Dict, Tuple
from enum import Enum


import argparse
import datetime
import json
import logging
import pickle
import random


import gymnasium as gym
import numpy
import yaml


from scipy import stats


from simulator.job import *
from simulator.charger import *
from simulator.demand import *
from simulator.region import *
from simulator.vehicle import *
from simulator.pricing import PricingUtility



random.seed(0)
numpy.random.seed(0)


class TaxiFleetSimulator(gym.Env):
    """Taxi fleet simulator.

    Args:
        seed: seed value for random number generator
        config: configuration dictionary, (see config.yaml for details.)
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config

    def _get_obs(self) -> numpy.array:
        """Get an observation from the environment."""
        # obs = numpy.zeros((len(self.fleet), 3))
        # for idx, v in enumerate(self.fleet):
        #     obs[idx, 0] = v.battery.actual_capacity / v.battery.initial_capacity
        #     obs[idx, 1] = v.battery.soc
        #     obs[idx, 2] = self.current_price # Broadcast price to all vehicles
    
        obs = numpy.zeros((len(self.fleet), 4))
        for idx, v in enumerate(self.fleet):
            obs[idx, 0] = v.battery.actual_capacity / v.battery.initial_capacity
            obs[idx, 1] = v.battery.soc
            obs[idx, 2] = self.current_price
            obs[idx, 3] = self.current_grid_power # Let the agent see the grid!

        return obs
    

    def reset(self, seed: int = None) -> Tuple[numpy.array, Dict]:
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes

        Returns:
            tuple: (obeservation, info) for initial state
        """
        super().reset(seed=seed)

        # 1. Initialize Time, Region, and Pricing first
        self.dt = float(self.config['delta t'])
        self.t = datetime.datetime.strptime(self.config['start t'], '%Y/%m/%d %H:%M:%S')
        self.t_max = datetime.datetime.strptime(self.config['end t'], '%Y/%m/%d %H:%M:%S')
        self.T_a = 25 

        self.region = CyclicZoneGraph(self.config['city']) 

        self.pricing = PricingUtility(
            city='chicago' if 'chicago' in self.config['city'].lower() else 'nyc',
            start_date=self.config['start t'].split(' ')[0].replace('/', '-'),
            end_date=self.config['end t'].split(' ')[0].replace('/', '-')
        )
        self.current_price = self.pricing.get_price(self.t)

        self.demand = ReplayDemand(self.config['demand'], self.region)
        self.demand.seek(self.t)
        self.arrived = self.demand.tick(self.dt)
        self.assigned = set()
        self.inprogress = set()
        self.rejected = 0
        self.completed = 0
        self.failed = 0
        self.current_grid_power = 0.0

        # 2. Initialize Fleet FIRST (so self.fleet exists!)
        self.fleet = []
        for vehicle in range(self.config['fleet']['size']):
            self.fleet.append(Vehicle(
                model=self.config['fleet']['vehicle'],
                battery=self.config['fleet']['battery model'],
                location=CyclicZoneGraphLocation(random.choice(list(self.region.map.keys())), self.region),
                vid=vehicle
            ))

        # 3. Initialize Charging Network
        self.charging_network = []
        for station in self.config['charging stations']:
            self.charging_network.append(ChargeStation(
                location = CyclicZoneGraphLocation(station['location'], self.region),
                ports = [ChargePort(station['max port power'], station['efficiency']) for port in range(station['ports'])],
                P_max = station['max total power'],
            ))

        # 4. Now initialize State and Action Spaces (safe to use len(self.fleet))
        self.observation_space = gym.spaces.Box(0, 1, shape=(len(self.fleet), 4))
        
        self.enable_v2g = self.config.get('enable_v2g', False) 
        self.c_max = self.config['charging stations'][0]['max port power'] if self.config['charging stations'] else 50.0
        
        low_action = numpy.zeros((len(self.fleet), 2))
        high_action = numpy.ones((len(self.fleet), 2))
        if self.enable_v2g:
            low_action[:, 1] = -self.c_max 
        high_action[:, 1] = self.c_max
        self.action_space = gym.spaces.Box(low=low_action, high=high_action, shape=(len(self.fleet), 2))
        self.step_count = 0


        # # Initialize Time
        # self.dt = float(self.config['delta t'])
        # self.t = datetime.datetime.strptime(self.config['start t'], '%Y/%m/%d %H:%M:%S')
        # self.t_max = datetime.datetime.strptime(self.config['end t'], '%Y/%m/%d %H:%M:%S')
        # self.T_a = 25 #TODO Weather model

        # # Load Map
        # self.region = CyclicZoneGraph(self.config['city']) 

        # # Load Demand
        # self.demand = ReplayDemand(self.config['demand'], self.region)
        # self.demand.seek(self.t)
        # self.arrived = self.demand.tick(self.dt)
        # self.assigned = set()
        # self.inprogress = set()
        # self.rejected = 0
        # self.completed = 0
        # self.failed = 0

        # self.observation_space = gym.spaces.Box(0, 1, shape=(len(self.fleet), 3))

        # self.pricing = PricingUtility(
        #     city='chicago' if 'chicago' in self.config['city'].lower() else 'nyc',
        #     start_date=self.config['start t'].split(' ')[0].replace('/', '-'),
        #     end_date=self.config['end t'].split(' ')[0].replace('/', '-')
        # )
        # self.current_price = self.pricing.get_price(self.t)

        # # Initialize Fleet
        # self.fleet = []
        # for vehicle in range(self.config['fleet']['size']):
        #     self.fleet.append(Vehicle(
        #         model=self.config['fleet']['vehicle'],
        #         battery=self.config['fleet']['battery model'],
        #         location=CyclicZoneGraphLocation(random.choice(list(self.region.map.keys())), self.region),
        #         vid=vehicle
        #     ))

        # # Initialize Charging Network
        # self.charging_network = []
        # for station in self.config['charging stations']:
        #     self.charging_network.append(ChargeStation(
        #         location = CyclicZoneGraphLocation(station['location'], self.region),
        #         ports = [ChargePort(station['max port power'], station['efficiency']) for port in range(station['ports'])],
        #         P_max = station['max total power'],
        #     ))

        # # Initialize State and Action Spaces
        # self.observation_space = gym.spaces.Box(0,1, shape=(len(self.fleet), 2))
        
        # self.enable_v2g = self.config.get('enable_v2g', False) # Toggle V2G from config
        # self.c_max = self.config['charging stations'][0]['max port power'] if self.config['charging stations'] else 50.0
        
        # low_action = numpy.zeros((len(self.fleet), 2))
        # high_action = numpy.ones((len(self.fleet), 2))
        # if self.enable_v2g:
        #     low_action[:, 1] = -self.c_max # Allow negative charge rate for V2G
        # high_action[:, 1] = self.c_max
        # self.action_space = gym.spaces.Box(low=low_action, high=high_action, shape=(len(self.fleet), 2))
        # self.step_count = 0

        # Global state information
        info = {}
        info['arrived'] = [j.to_dict() for j in self.arrived]
        info['assigned'] = [j.to_dict() for j in self.assigned]
        info['completed'] = self.completed
        info['rejected'] = self.rejected
        info['inprogress'] = [j.to_dict() for j in self.inprogress]
        info['failed'] = self.failed
        info['charging_network'] = [s.to_dict() for s in self.charging_network]
        info['fleet'] = [v.to_dict() for v in self.fleet]

        return self._get_obs(), info

    def get_closest_charger(self, vehicle: Vehicle) -> ChargeStation:
        """
        Get the closest charger to a <vehicle>.
        """
        distances = []
        for charger in self.charging_network:
            d, t = vehicle.location.to(charger.location)
            distances.append(d)
        return self.charging_network[distances.index(min(distances))]

    # def get_closest_job(self, vehicle: Vehicle) -> Job:
    #     """
    #     Get the closest job to <vehicle> that is not inprogress or expired.
    #     """
    #     closest_job = None
    #     distance = float('inf')
    #     for job in self.arrived:
    #         d, t = vehicle.location.to(job.pickup_location)
    #         #if d == float('inf'):
    #         #    print(job.pickup_location.region.map[1])
    #         if d < distance:
    #             distance = d
    #             closest_job = job
    #     return closest_job
    
    def get_closest_job(self, vehicle: Vehicle) -> Job:
        """
        Get the closest job to <vehicle> that is not inprogress or expired.
        """
        closest_job = None
        distance = float('inf')
        for job in self.arrived:
            d, t = vehicle.location.to(job.pickup_location)
            # Ensure distance is valid and not None before comparison
            if d is not None and d < distance:
                distance = d
                closest_job = job
        return closest_job

    def step(self, action: numpy.array) -> Tuple[numpy.array, float, bool, bool, Dict]:
        """Execute one timestep within the environment.

        Args:
            action: The action to take
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        # First update vehicle statuses
        eta = 0.90 # Round-trip efficiency loss
        
        self.current_grid_power = 0.0 # Track exact grid power

        # for idx in range(len(self.fleet)):
        #     charge_flag, c_v = action[idx,0], action[idx,1]
        #     if not self.enable_v2g:
        #         c_v = max(0.0, c_v)

        #     if charge_flag > 0.5 and self.fleet[idx].status in [VehicleStatus.IDLE, VehicleStatus.CHARGING, VehicleStatus.TOCHARGE]:
        #         if c_v >= 0:
        #             self.fleet[idx].charge(self.get_closest_charger(self.fleet[idx]), c_v)
        #         else:
        #             # Update SoC directly for discharging with efficiency loss
        #             energy_change = (c_v / eta * (self.dt / 3600.0)) / self.fleet[idx].battery.initial_capacity
        #             self.fleet[idx].battery.soc = max(0.0, self.fleet[idx].battery.soc + energy_change)
        #             self.current_grid_power += c_v # Record negative power from V2G
        #     elif len(self.arrived) > 0 and self.fleet[idx].status in [VehicleStatus.IDLE, VehicleStatus.CHARGING, VehicleStatus.TOCHARGE]:
        #         self.fleet[idx].service_demand(self.get_closest_job(self.fleet[idx]))

        for idx in range(len(self.fleet)):
            # If your action space has 2 columns, let column 1 dictate charging/discharging power directly,
            # or combine them so the agent doesn't get stuck behind a strict boolean gate.
            c_v = action[idx, 1]
            if not self.enable_v2g:
                c_v = max(0.0, c_v)

            # Allow vehicles to charge/discharge if they are idle or already charging, 
            # or let them service demand if no charging action is strongly commanded.
            if abs(c_v) > 1.0 and self.fleet[idx].status in [VehicleStatus.IDLE, VehicleStatus.CHARGING, VehicleStatus.TOCHARGE]:
                # print("Ability to make choice")
                if c_v >= 0:
                    target_charger = self.get_closest_charger(self.fleet[idx])
                    if self.fleet[idx].charger != target_charger or getattr(self.fleet[idx], 'preferred_rate', 0) != c_v:
                        self.fleet[idx].charge(target_charger, c_v)
                    # self.fleet[idx].charge(self.get_closest_charger(self.fleet[idx]), c_v)
                    # self.current_grid_power += c_v
                else:
                    energy_change = (c_v / eta * (self.dt / 3600.0)) / self.fleet[idx].battery.initial_capacity
                    self.fleet[idx].battery.soc = max(0.0, self.fleet[idx].battery.soc + energy_change)
                    self.current_grid_power += c_v # Negative power from V2G

            elif len(self.arrived) > 0 and self.fleet[idx].status in [VehicleStatus.IDLE, VehicleStatus.CHARGING, VehicleStatus.TOCHARGE]:
                self.fleet[idx].service_demand(self.get_closest_job(self.fleet[idx]))

        # Update fleet
        for vehicle in self.fleet:
            vehicle.tick(self.dt, {'T_a': self.T_a}) # TODO: Check conditions

        # Update charging vehicles
        for charger in self.charging_network:
            charger.tick(self.fleet, self.dt, self.T_a)
            for port in charger.ports:
                if port.vehicle is not None:
                    self.current_grid_power += port.P_t

        # Get new arrivals
        self.arrived = self.arrived | self.demand.tick(self.dt)

        # Update jobs in progress
        to_completed = set()
        to_failed = set()
        for job in self.inprogress:
            if job.status == JobStatus.COMPLETE:
                to_completed = to_completed.union({job})
            elif job.status == JobStatus.FAILED:
                to_failed = to_failed.union({job})
        self.inprogress = self.inprogress - to_completed - to_failed
        self.completed += len(to_completed)
        self.failed += len(to_failed)

        # Update assigned jobs
        to_inprogress = set()
        to_failed = set()
        for job in self.assigned:
            if job.status == JobStatus.INPROGRESS:
                to_inprogress = to_inprogress.union({job})
            elif job.status == JobStatus.FAILED:
                to_failed = to_failed.union({job})
        self.assigned = self.assigned - to_inprogress - to_failed
        self.failed += len(to_failed)
        self.inprogress = self.inprogress.union(to_inprogress)

        # Update arrived jobs
        to_assigned = set()
        to_rejected = set()
        for job in self.arrived:
            job.tick(self.dt)
            if job.status == JobStatus.ASSIGNED:
                to_assigned = to_assigned.union({job})
            elif job.status == JobStatus.REJECTED:
                to_rejected = to_rejected.union({job})
            elif job.status == JobStatus.INPROGRESS:
                to_inprogress = to_inprogress.union({job})
        self.arrived = self.arrived - to_assigned - to_rejected - to_inprogress
        self.assigned = self.assigned.union(to_assigned)
        self.inprogress = self.inprogress.union(to_inprogress)
        self.rejected += len(to_rejected)

        # Update time
        self.t = self.t + datetime.timedelta(seconds=self.dt)

        # Updating Price
        self.current_price = self.pricing.get_price(self.t)
        self.step_count += 1
        
        print(self.t)

        # Calculate info
        info = {}
        info['arrived'] = [j.to_dict() for j in self.arrived]
        info['assigned'] = [j.to_dict() for j in self.assigned]
        info['completed'] = self.completed
        info['rejected'] = self.rejected
        info['inprogress'] = [j.to_dict() for j in self.inprogress]
        info['failed'] = self.failed
        info['charging_network'] = [s.to_dict() for s in self.charging_network]
        info['fleet'] = [v.to_dict() for v in self.fleet]
        info['total_grid_power'] = self.current_grid_power
        
        # Calculate reward
        # TODO: specify as lambda

        
        # ALPHA = 1.0
        # BETA = 1.0
        # #reward = sum([v.battery.soc for v in self.fleet]) + LAMBDA * sum([v.battery.actual_capacity / v.battery.initial_capacity for v in self.fleet])
        # reward = self.completed + ALPHA * sum([v.battery.actual_capacity / v.battery.initial_capacity for v in self.fleet]) # - BETA * sum([1 if v.status == VehicleStatus.RECOVERY else - for v in self.fleet])

        # Calculate reward
        ALPHA = 1.0  # Weight for battery health
        BETA = 0.5   # Weight for V2B revenue
        LAMBDA = 1.0 # Lagrangian multiplier for peak constraint violation

        # 1. Base Taxi Revenue & Health
        step_completed = len(to_completed)
        reward = step_completed + ALPHA * sum([v.battery.actual_capacity / v.battery.initial_capacity for v in self.fleet])

        # 2. V2B Arbitrage Revenue and Building Load Calculation
        BUILDING_BASELINE_LOAD = 100.0 # kW (Static for now, can be randomized later)
        BUILDING_PEAK_LIMIT = 500.0 # kW threshold before massive fees

        net_building_load = BUILDING_BASELINE_LOAD + self.current_grid_power
        v2b_revenue = 0.0

        for idx, vehicle in enumerate(self.fleet):
            charge_action = action[idx, 1]
            if vehicle.status in [VehicleStatus.IDLE, VehicleStatus.CHARGING] and charge_action != 0:
                # If discharging (negative), building load decreases. If charging (positive), load increases.
                net_building_load += charge_action
                
                # If discharging, generate revenue based on current price
                if charge_action < 0:
                    # kWh = kW * hours
                    kwh_discharged = abs(charge_action) * (self.dt / 3600.0)
                    v2b_revenue += kwh_discharged * self.current_price

        # Add arbitrage profit to the reward
        reward += BETA * v2b_revenue

        # 3. Micro-Grid Peak Penalty (Lagrangian Constraint)
        if net_building_load > BUILDING_PEAK_LIMIT:
            violation_magnitude = net_building_load - BUILDING_PEAK_LIMIT
            # Quadratic penalty forces the agent to strictly avoid crossing the boundary
            reward -= LAMBDA * (violation_magnitude ** 2)

        # V2G Idle Reward Tweak: positive reward for discharging while idle
        v2g_bonus = 0.05
        for idx, vehicle in enumerate(self.fleet):
            if vehicle.status == VehicleStatus.IDLE and action[idx, 1] < 0:
                reward += v2g_bonus * abs(action[idx, 1])

        return (
            self._get_obs(),
            reward,
            True if self.t >= self.t_max else False,
            True if self.step_count > 1000 else False,
            info
        )


