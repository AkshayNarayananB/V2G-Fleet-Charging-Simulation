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
import numpy as np
import pandas as pd
import yaml

from scipy import stats

from simulator.job import Job, JobStatus 
from simulator.charger import *
from simulator.demand import *
from simulator.region import *
from simulator.vehicle import *
from simulator.pricing import PricingUtility

random.seed(0)
np.random.seed(0)

class TaxiFleetSimulator(gym.Env):
    """Taxi fleet simulator."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        
        self.reward_type = self.config.get('reward_type', 'economics').lower()
        self.lagrangian_multiplier = 10.0 
        self.enable_v2g = self.config.get('enable_v2g', True) 
        
        fleet_size = self.config['fleet']['size']
        self.c_max = self.config['charging stations'][0]['max port power'] if self.config.get('charging stations') else 50.0

        obs_dim = (fleet_size * 2) + 5 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        low_action = np.full((fleet_size, 2), -1.0, dtype=np.float32)
        high_action = np.full((fleet_size, 2), 1.0, dtype=np.float32)
        if not self.enable_v2g:
            low_action[:, 1] = 0.0
            
        self.action_space = gym.spaces.Box(low=low_action, high=high_action, shape=(fleet_size, 2), dtype=np.float32)
        
    def _get_baseline_load(self, timestamp: datetime.datetime) -> float:
        minute_floor = (timestamp.minute // 15) * 15
        lookup_time = timestamp.replace(minute=minute_floor, second=0, microsecond=0)
        
        try:
            return float(self.building_data.loc[lookup_time, 'Aggregate_kW'])
        except KeyError:
            return 350.0 

    def _get_obs(self) -> np.array:
        current_hour = self.t.hour + (self.t.minute / 60.0)
        
        obs = []
        for v in self.fleet:
            obs.append(v.battery.actual_capacity / v.battery.initial_capacity)
            obs.append(v.battery.soc)
            
        obs.append(current_hour)
        obs.append(self._get_baseline_load(self.t))
        obs.append(self._get_baseline_load(self.t + datetime.timedelta(minutes=15)))
        obs.append(self._get_baseline_load(self.t + datetime.timedelta(minutes=30)))
        obs.append(self._get_baseline_load(self.t + datetime.timedelta(minutes=45)))
        
        return np.array(obs, dtype=np.float32)
    
    def get_closest_charger_from_loc(self, loc: Location) -> Tuple[ChargeStation, float]:
        best_charger = self.charging_network[0]
        min_d = float('inf')
        for charger in self.charging_network:
            if loc == charger.location:
                return charger, 0.0
            d, _ = loc.to(charger.location)
            if d is not None and d < min_d:
                min_d = d
                best_charger = charger
        return best_charger, min_d

    def get_closest_charger(self, vehicle: Vehicle) -> ChargeStation:
        charger, _ = self.get_closest_charger_from_loc(vehicle.location)
        return charger
    
    def get_closest_job(self, vehicle: Vehicle) -> Job:
        closest_job = None
        distance = float('inf')
        for job in self.arrived:
            if job.status != JobStatus.ARRIVED:
                continue
            d, t = vehicle.location.to(job.pickup_location)
            if d is not None and d < distance:
                distance = d
                closest_job = job
        return closest_job

    def _soc_needed_for_distance(self, vehicle: Vehicle, dist_km: float) -> float:
        """Exact SoC fraction needed to cover dist_km (matches vehicle.py dW calculation)."""
        if dist_km is None or dist_km == float('inf'):
            return 1.0
        dW = dist_km * (vehicle.efficiency / 100.0)
        return dW / max(1.0, vehicle.battery.actual_capacity)

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.array, Dict]:
        super().reset(seed=seed)

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

        try:
            self.building_data = pd.read_csv('data/nrel_building_loads.csv')
            self.building_data['timestamp'] = pd.to_datetime(self.building_data['timestamp'])
            self.building_data.set_index('timestamp', inplace=True)

            cluster_mix = self.config.get('microgrid_cluster', {'Office_kW': 2.0})
            self.building_data['Aggregate_kW'] = 0.0
            for bldg_type, multiplier in cluster_mix.items():
                if bldg_type in self.building_data.columns:
                    self.building_data['Aggregate_kW'] += self.building_data[bldg_type] * float(multiplier)
        except Exception as e:
            print(f"Warning: Could not load NREL building loads. Using fallback baseline. ({e})")
            self.building_data = pd.DataFrame(columns=['Aggregate_kW'])

        self.demand = ReplayDemand(self.config['demand'], self.region)
        self.demand.seek(self.t)
        self.arrived = self.demand.tick(self.dt)
        self.assigned = set()
        self.inprogress = set()
        self.rejected = 0
        self.completed = 0
        self.failed = 0
        self.current_grid_power = 0.0

        valid_zones = list(self.region.map.keys())
        cluster_zones = [z for z in [161, 162, 163, 170, 236, 237, 141, 142, 143, 239, 140, 137, 107, 224, 229] if z in valid_zones]
        spawn_zones = cluster_zones if len(cluster_zones) > 0 else valid_zones

        self.fleet = []
        for vehicle in range(self.config['fleet']['size']):
            self.fleet.append(Vehicle(
                model=self.config['fleet']['vehicle'],
                battery=self.config['fleet']['battery model'],
                location=CyclicZoneGraphLocation(random.choice(spawn_zones), self.region),
                vid=vehicle
            ))

        self.charging_network = []
        for idx, station in enumerate(self.config['charging stations']):
            loc_id = station['location'] if station['location'] in spawn_zones else spawn_zones[idx % len(spawn_zones)]
            self.charging_network.append(ChargeStation(
                location = CyclicZoneGraphLocation(loc_id, self.region),
                ports = [ChargePort(station['max port power'], station['efficiency']) for port in range(station['ports'])],
                P_max = station['max total power'],
            ))

        self.step_count = 0
        self.prev_capacity = sum([v.battery.actual_capacity for v in self.fleet])

        info = {
            'arrived': [j.to_dict() for j in self.arrived],
            'assigned': [j.to_dict() for j in self.assigned],
            'completed': self.completed,
            'rejected': self.rejected,
            'inprogress': [j.to_dict() for j in self.inprogress],
            'failed': self.failed,
            'charging_network': [s.to_dict() for s in self.charging_network],
            'fleet': [v.to_dict() for v in self.fleet],
            'step_completed': 0,
            'skipped_rides': 0,
            'V2B_Revenue': 0.0,
            'peak_violation': 0.0,
            'total_grid_power': 0.0,
            'current_price': self.current_price,
            'net_building_load': self._get_baseline_load(self.t)
        }

        return self._get_obs(), info

    # def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, Dict]:
    #     eta = 0.90 
    #     self.current_grid_power = 0.0 
    #     step_v2b_revenue = 0.0
    #     manual_v2g_power = 0.0

    #     BUILDING_PEAK_LIMIT = 500.0
    #     current_bldg_load = self._get_baseline_load(self.t)
    #     upcoming_bldg_load = max(
    #         current_bldg_load,
    #         self._get_baseline_load(self.t + datetime.timedelta(minutes=15)),
    #         self._get_baseline_load(self.t + datetime.timedelta(minutes=30))
    #     )
    #     peak_active = current_bldg_load > BUILDING_PEAK_LIMIT
    #     peak_anticipated = upcoming_bldg_load > BUILDING_PEAK_LIMIT

    #     # --- 1. AUTO-UNPLUG FULL CARS & PRIORITY PORT PREEMPTION ---
    #     for station in self.charging_network:
    #         cars_at_station = []
    #         for v in self.fleet:
    #             if v.charger == station and v.status in [VehicleStatus.CHARGING, VehicleStatus.DISCHARGING]:
    #                 if v.status == VehicleStatus.CHARGING and v.battery.soc >= 0.90:
    #                     station.disconnect(v.vid)
    #                     v.charger = None
    #                     v.status = VehicleStatus.IDLE
    #                     v.preferred_rate = 0.0
    #                 else:
    #                     cars_at_station.append(v)
            
    #         if len(cars_at_station) > len(station.ports):
    #             def port_priority(veh):
    #                 if veh.status == VehicleStatus.CHARGING:
    #                     return (0, veh.battery.soc)
    #                 else:
    #                     return (1, -veh.battery.soc)

    #             cars_at_station.sort(key=port_priority)
    #             for v in cars_at_station[len(station.ports):]:
    #                 station.disconnect(v.vid)
    #                 v.charger = None
    #                 v.status = VehicleStatus.IDLE
    #                 v.preferred_rate = 0.0

    #     # --- 2. VEHICLE DECISION LOOP ---
    #     for idx in range(len(self.fleet)):
    #         v = self.fleet[idx]

    #         # Busy cars on a trip or in recovery cannot be interrupted
    #         if v.status in [VehicleStatus.ONJOB, VehicleStatus.TOPICKUP, VehicleStatus.RECOVERY, VehicleStatus.TOLOC]:
    #             continue

    #         dispatch_intent = float(action[idx, 0])
    #         c_v_norm = float(action[idx, 1])
            
    #         if c_v_norm > 0.05:
    #             actual_kw = ((c_v_norm - 0.05) / 0.95) * self.c_max
    #         elif c_v_norm < -0.05 and self.enable_v2g:
    #             actual_kw = ((c_v_norm + 0.05) / 0.95) * self.c_max
    #         else:
    #             actual_kw = 0.0

    #         target_charger, d_to_charger = self.get_closest_charger_from_loc(v.location)
    #         at_charger = (v.status in [VehicleStatus.CHARGING, VehicleStatus.DISCHARGING]) or (v.location == target_charger.location)
    #         soc_to_charger = self._soc_needed_for_distance(v, d_to_charger) + 0.10

    #         # Check exact ride feasibility (unclamped so cars NEVER die mid-ride)
    #         closest_job = self.get_closest_job(v) if len(self.arrived) > 0 else None
    #         can_take_ride = False
    #         if closest_job is not None:
    #             d_pickup, _ = v.location.to(closest_job.pickup_location)
    #             d_ride, _ = closest_job.pickup_location.to(closest_job.dropoff_location)
    #             _, d_dropoff_to_chg = self.get_closest_charger_from_loc(closest_job.dropoff_location)
                
    #             if d_pickup is not None and d_ride is not None:
    #                 required_ride_soc = self._soc_needed_for_distance(v, d_pickup + d_ride + d_dropoff_to_chg) + 0.15
    #                 if v.battery.soc >= max(0.35, required_ride_soc):
    #                     can_take_ride = True

    #         # RULE 1: Critical Survival - Must charge immediately
    #         if v.battery.soc <= max(0.30, soc_to_charger):
    #             v.charge(target_charger, self.c_max)
    #             if at_charger:
    #                 v.status = VehicleStatus.CHARGING
    #             continue

    #         # RULE 2: V2B Peak Shaving - Discharge when peak is active/anticipated or agent commands V2G
    #         wants_v2g = (actual_kw < -2.0) or (peak_anticipated and not can_take_ride) or (peak_active and v.battery.soc > 0.50)
    #         if self.enable_v2g and wants_v2g and v.battery.soc > 0.35:
    #             discharge_kw = actual_kw if actual_kw < -5.0 else -self.c_max
    #             if not at_charger:
    #                 v.charge(target_charger, discharge_kw)
    #             else:
    #                 if v.charger is not None and v.charger != target_charger:
    #                     v.charger.disconnect(v.vid)
    #                 v.charger = target_charger
    #                 v.preferred_rate = discharge_kw
    #                 v.status = VehicleStatus.DISCHARGING
                    
    #                 # Direct V2B energy & revenue calculation
    #                 soc_drop = min(v.battery.soc - 0.30, (abs(discharge_kw) * (self.dt / 3600.0) / eta) / v.battery.initial_capacity)
    #                 if soc_drop > 0:
    #                     v.battery.soc -= soc_drop
    #                     discharged_kwh = soc_drop * v.battery.initial_capacity * eta
    #                     effective_grid_kw = -discharged_kwh / (self.dt / 3600.0)
                        
    #                     manual_v2g_power += effective_grid_kw
    #                     step_v2b_revenue += discharged_kwh * max(0.50, self.current_price)
    #             continue

    #         # RULE 3: Complete Rides (If feasible)
    #         if can_take_ride and (dispatch_intent >= -0.2 or v.battery.soc > 0.50):
    #             if v.status in [VehicleStatus.CHARGING, VehicleStatus.DISCHARGING] and v.charger is not None:
    #                 v.charger.disconnect(v.vid)
    #                 v.charger = None
    #             v.service_demand(closest_job)
    #             self.arrived.remove(closest_job)
    #             self.assigned.add(closest_job)
    #             continue

    #         # RULE 4: Cannot take a ride & Battery < 85% -> Go Charge
    #         if (not can_take_ride and v.battery.soc < 0.85) or (actual_kw > 0.0 and v.battery.soc < 0.90):
    #             charge_kw = actual_kw if actual_kw > 10.0 else self.c_max
    #             if peak_active and v.battery.soc > 0.40:
    #                 if v.status in [VehicleStatus.CHARGING, VehicleStatus.DISCHARGING] and v.charger is not None:
    #                     v.charger.disconnect(v.vid)
    #                     v.charger = None
    #                 v.status = VehicleStatus.IDLE
    #             else:
    #                 v.charge(target_charger, charge_kw)
    #                 if at_charger:
    #                     v.status = VehicleStatus.CHARGING
    #             continue

    #         # RULE 5: Otherwise stay IDLE
    #         if v.status in [VehicleStatus.CHARGING, VehicleStatus.DISCHARGING]:
    #             if v.charger is not None:
    #                 v.charger.disconnect(v.vid)
    #                 v.charger = None
    #             v.status = VehicleStatus.IDLE

    #     # --- 3. PHYSICS & DEMAND TICKS ---
    #     for vehicle in self.fleet:
    #         vehicle.tick(self.dt, {'T_a': self.T_a}) 

    #     for charger in self.charging_network:
    #         charger.tick(self.fleet, self.dt, self.T_a)
    #         for port in charger.ports:
    #             if port.vehicle is not None and port.P_t > 0:
    #                 self.current_grid_power += port.P_t

    #     self.current_grid_power += manual_v2g_power
    #     self.arrived = self.arrived | self.demand.tick(self.dt)

    #     to_completed = set()
    #     to_failed = set()
    #     for job in self.inprogress:
    #         if job.status == JobStatus.COMPLETE:
    #             to_completed = to_completed.union({job})
    #         elif job.status == JobStatus.FAILED:
    #             to_failed = to_failed.union({job})
    #     self.inprogress = self.inprogress - to_completed - to_failed
    #     step_completed = len(to_completed)
    #     self.completed += step_completed
    #     self.failed += len(to_failed)

    #     to_inprogress = set()
    #     to_failed_assigned = set()
    #     for job in self.assigned:
    #         if job.status == JobStatus.INPROGRESS:
    #             to_inprogress = to_inprogress.union({job})
    #         elif job.status == JobStatus.FAILED:
    #             to_failed_assigned = to_failed_assigned.union({job})
    #     self.assigned = self.assigned - to_inprogress - to_failed_assigned
    #     self.failed += len(to_failed_assigned)
    #     self.inprogress = self.inprogress.union(to_inprogress)

    #     to_assigned = set()
    #     to_rejected = set()
    #     to_inprogress_arrived = set()
    #     for job in self.arrived:
    #         job.tick(self.dt)
    #         if job.status == JobStatus.ASSIGNED:
    #             to_assigned = to_assigned.union({job})
    #         elif job.status == JobStatus.REJECTED:
    #             to_rejected = to_rejected.union({job})
    #         elif job.status == JobStatus.INPROGRESS:
    #             to_inprogress_arrived = to_inprogress_arrived.union({job})
    #     self.arrived = self.arrived - to_assigned - to_rejected - to_inprogress_arrived
    #     self.assigned = self.assigned.union(to_assigned)
    #     self.inprogress = self.inprogress.union(to_inprogress_arrived)
    #     step_rejected = len(to_rejected)
    #     self.rejected += step_rejected

    #     self.t = self.t + datetime.timedelta(seconds=self.dt)
    #     self.current_price = self.pricing.get_price(self.t)
    #     self.step_count += 1

    #     current_capacity = sum([v.battery.actual_capacity for v in self.fleet])
    #     step_soh_loss_kwh = max(0.0, self.prev_capacity - current_capacity)
    #     self.prev_capacity = current_capacity

    #     # --- 4. REWARD CALCULATION ---
    #     net_building_load = current_bldg_load + self.current_grid_power
    #     violation = max(0.0, net_building_load - BUILDING_PEAK_LIMIT)
    #     peak_shaved_kw = max(0.0, current_bldg_load - net_building_load) if peak_active else 0.0

    #     BATTERY_COST_PER_KWH = 135.0
        
    #     if self.reward_type == "operations":
    #         MOBILITY_REWARD = 250.0
    #         reward = (step_completed * MOBILITY_REWARD) + (peak_shaved_kw * 25.0) + (step_v2b_revenue * 10.0)
    #         reward -= (step_soh_loss_kwh * BATTERY_COST_PER_KWH)
    #         reward -= self.lagrangian_multiplier * violation

    #     elif self.reward_type == "economics":
    #         MOBILITY_WEIGHT = 15.0
    #         fare_revenue = sum([job.fare for job in to_completed])
    #         reward = (fare_revenue * MOBILITY_WEIGHT) + (peak_shaved_kw * 25.0) + (step_v2b_revenue * 100.0)
    #         reward -= (step_soh_loss_kwh * BATTERY_COST_PER_KWH)
    #         reward -= self.lagrangian_multiplier * violation
    #     else:
    #         reward = 0.0
            
    #     for v in self.fleet:
    #         if v.status == VehicleStatus.RECOVERY:
    #             reward -= 1000.0 

    #     info = {
    #         'arrived': [j.to_dict() for j in self.arrived],
    #         'assigned': [j.to_dict() for j in self.assigned],
    #         'completed': self.completed,
    #         'step_completed': step_completed,
    #         'rejected': self.rejected,
    #         'skipped_rides': step_rejected,
    #         'inprogress': [j.to_dict() for j in self.inprogress],
    #         'failed': self.failed,
    #         'charging_network': [s.to_dict() for s in self.charging_network],
    #         'fleet': [v.to_dict() for v in self.fleet],
    #         'total_grid_power': self.current_grid_power,
    #         'net_building_load': net_building_load,
    #         'V2B_Revenue': step_v2b_revenue, 
    #         'current_price': self.current_price,
    #         'peak_violation': violation 
    #     }

    #     return (
    #         self._get_obs(),
    #         reward,
    #         True if self.t >= self.t_max else False,
    #         False,
    #         info
    #     )

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, Dict]:
        eta = 0.90 
        self.current_grid_power = 0.0 
        step_v2b_revenue = 0.0
        manual_v2g_power = 0.0

        BUILDING_PEAK_LIMIT = 500.0
        current_bldg_load = self._get_baseline_load(self.t)
        upcoming_bldg_load = max(
            current_bldg_load,
            self._get_baseline_load(self.t + datetime.timedelta(minutes=15)),
            self._get_baseline_load(self.t + datetime.timedelta(minutes=30))
        )
        peak_active = current_bldg_load > BUILDING_PEAK_LIMIT
        peak_anticipated = upcoming_bldg_load > BUILDING_PEAK_LIMIT

        # Exact kW needed right now to shave the building peak down to 500 kW
        remaining_v2b_needed_kw = max(0.0, current_bldg_load - BUILDING_PEAK_LIMIT)

        # Convert wholesale $/MWh price to $/kWh (if PricingUtility is in $/MWh)
        raw_price = float(self.current_price) if self.current_price is not None else 0.20
        price_per_kwh = (raw_price / 1000.0) if raw_price > 2.0 else max(0.15, raw_price)

        # --- 1. AUTO-UNPLUG FULL CARS & PRIORITY PORT PREEMPTION ---
        for station in self.charging_network:
            cars_at_station = []
            for v in self.fleet:
                if v.charger == station and v.status in [VehicleStatus.CHARGING, VehicleStatus.DISCHARGING]:
                    if v.status == VehicleStatus.CHARGING and v.battery.soc >= 0.90:
                        station.disconnect(v.vid)
                        v.charger = None
                        v.status = VehicleStatus.IDLE
                        v.preferred_rate = 0.0
                    else:
                        cars_at_station.append(v)
            
            if len(cars_at_station) > len(station.ports):
                def port_priority(veh):
                    if veh.status == VehicleStatus.CHARGING:
                        return (0, veh.battery.soc)
                    else:
                        return (1, -veh.battery.soc)

                cars_at_station.sort(key=port_priority)
                for v in cars_at_station[len(station.ports):]:
                    station.disconnect(v.vid)
                    v.charger = None
                    v.status = VehicleStatus.IDLE
                    v.preferred_rate = 0.0

        # --- 2. VEHICLE DECISION LOOP ---
        for idx in range(len(self.fleet)):
            v = self.fleet[idx]

            # Busy cars on a trip or in recovery cannot be interrupted
            if v.status in [VehicleStatus.ONJOB, VehicleStatus.TOPICKUP, VehicleStatus.RECOVERY, VehicleStatus.TOLOC]:
                continue

            dispatch_intent = float(action[idx, 0])
            c_v_norm = float(action[idx, 1])
            
            if c_v_norm > 0.05:
                actual_kw = ((c_v_norm - 0.05) / 0.95) * self.c_max
            elif c_v_norm < -0.05 and self.enable_v2g:
                actual_kw = ((c_v_norm + 0.05) / 0.95) * self.c_max
            else:
                actual_kw = 0.0

            target_charger, d_to_charger = self.get_closest_charger_from_loc(v.location)
            at_charger = (v.status in [VehicleStatus.CHARGING, VehicleStatus.DISCHARGING]) or (v.location == target_charger.location)
            soc_to_charger = self._soc_needed_for_distance(v, d_to_charger) + 0.10

            # Check exact ride feasibility (unclamped so cars NEVER die mid-ride)
            closest_job = self.get_closest_job(v) if len(self.arrived) > 0 else None
            can_take_ride = False
            if closest_job is not None:
                d_pickup, _ = v.location.to(closest_job.pickup_location)
                d_ride, _ = closest_job.pickup_location.to(closest_job.dropoff_location)
                _, d_dropoff_to_chg = self.get_closest_charger_from_loc(closest_job.dropoff_location)
                
                if d_pickup is not None and d_ride is not None:
                    required_ride_soc = self._soc_needed_for_distance(v, d_pickup + d_ride + d_dropoff_to_chg) + 0.15
                    if v.battery.soc >= max(0.35, required_ride_soc):
                        can_take_ride = True

            # RULE 1: Critical Survival - Must charge immediately
            if v.battery.soc <= max(0.30, soc_to_charger):
                v.charge(target_charger, self.c_max)
                if at_charger:
                    v.status = VehicleStatus.CHARGING
                continue

            # # RULE 2: V2B Peak Shaving (Strictly capped to the building's excess load above 500 kW)
            # wants_v2g = (actual_kw < -2.0) or (peak_anticipated and not can_take_ride)
            # if self.enable_v2g and wants_v2g and v.battery.soc > 0.40:
            #     # Case 2A: Peak is anticipated soon, route to charger to prepare
            #     if not at_charger and peak_anticipated:
            #         v.charge(target_charger, -self.c_max)
            #         continue
            #     # Case 2B: At charger AND building currently needs peak shaving (remaining_v2b_needed_kw > 0)
            #     elif at_charger and peak_active and remaining_v2b_needed_kw > 0.1:
            #         req_discharge_kw = abs(actual_kw) if actual_kw < -5.0 else self.c_max
            #         # Clamp to what the building actually needs so net load never drops below 500 kW
            #         allocated_grid_kw = min(req_discharge_kw, remaining_v2b_needed_kw, self.c_max)
                    
            #         # Ensure battery doesn't drop below 40% SoC
            #         max_removable_kwh = max(0.0, (v.battery.soc - 0.40) * v.battery.actual_capacity)
            #         desired_battery_kwh = (allocated_grid_kw / eta) * (self.dt / 3600.0)
            #         actual_battery_kwh = min(max_removable_kwh, desired_battery_kwh)
                    
            #         if actual_battery_kwh > 0.01:
            #             if v.charger is not None and v.charger != target_charger:
            #                 v.charger.disconnect(v.vid)
            #             v.charger = target_charger
            #             v.preferred_rate = -allocated_grid_kw
            #             v.status = VehicleStatus.DISCHARGING
                        
            #             # Use native battery.discharge() so SOH degradation & internal states are accurate
            #             v.battery.discharge(actual_battery_kwh, self.dt, self.T_a)
                        
            #             delivered_kwh = actual_battery_kwh * eta
            #             effective_grid_kw = delivered_kwh / (self.dt / 3600.0)
                        
            #             manual_v2g_power -= effective_grid_kw
            #             remaining_v2b_needed_kw = max(0.0, remaining_v2b_needed_kw - effective_grid_kw)
            #             step_v2b_revenue += delivered_kwh * price_per_kwh
            #             continue

            # RULE 2: Agent-Driven V2B Peak Shaving (Controlled by PPO action `actual_kw < 0`)
            # If the agent commands discharge (actual_kw < -2.0) AND the building is over or approaching peak
            wants_v2g = (actual_kw < -2.0) and (peak_active or peak_anticipated)
            
            if self.enable_v2g and wants_v2g and v.battery.soc > 0.35 and remaining_v2b_needed_kw > 0.1:
                if not at_charger:
                    # Route to nearest charger to execute V2B
                    v.charge(target_charger, actual_kw)
                else:
                    allocated_grid_kw = min(abs(actual_kw), remaining_v2b_needed_kw, self.c_max)
                    
                    max_removable_kwh = max(0.0, (v.battery.soc - 0.30) * v.battery.actual_capacity)
                    desired_battery_kwh = (allocated_grid_kw / eta) * (self.dt / 3600.0)
                    actual_battery_kwh = min(max_removable_kwh, desired_battery_kwh)
                    
                    if actual_battery_kwh > 0.01:
                        if v.charger is not None and v.charger != target_charger:
                            v.charger.disconnect(v.vid)
                        v.charger = target_charger
                        v.preferred_rate = -allocated_grid_kw
                        v.status = VehicleStatus.DISCHARGING
                        
                        v.battery.discharge(actual_battery_kwh, self.dt, self.T_a)
                        
                        delivered_kwh = actual_battery_kwh * eta
                        effective_grid_kw = delivered_kwh / (self.dt / 3600.0)
                        
                        manual_v2g_power -= effective_grid_kw
                        remaining_v2b_needed_kw = max(0.0, remaining_v2b_needed_kw - effective_grid_kw)
                        
                        # Use dynamic electricity pricing (or retail feed-in tariff ~ $0.20 - $0.35/kWh)
                        step_v2b_revenue += delivered_kwh * max(0.25, price_per_kwh * 2.0)
                        continue


            # RULE 3: Complete Rides (If feasible)
            if can_take_ride and (dispatch_intent >= -0.2 or v.battery.soc > 0.50):
                if v.status in [VehicleStatus.CHARGING, VehicleStatus.DISCHARGING] and v.charger is not None:
                    v.charger.disconnect(v.vid)
                    v.charger = None
                v.service_demand(closest_job)
                self.arrived.remove(closest_job)
                self.assigned.add(closest_job)
                continue

            # RULE 4: Cannot take a ride & Battery < 85% -> Go Charge (if not violating peak)
            if (not can_take_ride and v.battery.soc < 0.85) or (actual_kw > 0.0 and v.battery.soc < 0.90):
                charge_kw = actual_kw if actual_kw > 10.0 else self.c_max
                if peak_active and v.battery.soc > 0.40:
                    if v.status in [VehicleStatus.CHARGING, VehicleStatus.DISCHARGING] and v.charger is not None:
                        v.charger.disconnect(v.vid)
                        v.charger = None
                    v.status = VehicleStatus.IDLE
                else:
                    v.charge(target_charger, charge_kw)
                    if at_charger:
                        v.status = VehicleStatus.CHARGING
                continue

            # RULE 5: Otherwise stay IDLE
            if v.status in [VehicleStatus.CHARGING, VehicleStatus.DISCHARGING]:
                if v.charger is not None:
                    v.charger.disconnect(v.vid)
                    v.charger = None
                v.status = VehicleStatus.IDLE

        # --- 3. PHYSICS & DEMAND TICKS ---
        for vehicle in self.fleet:
            vehicle.tick(self.dt, {'T_a': self.T_a}) 

        for charger in self.charging_network:
            charger.tick(self.fleet, self.dt, self.T_a)
            for port in charger.ports:
                if port.vehicle is not None and port.P_t > 0:
                    self.current_grid_power += port.P_t

        self.current_grid_power += manual_v2g_power
        self.arrived = self.arrived | self.demand.tick(self.dt)

        to_completed = set()
        to_failed = set()
        for job in self.inprogress:
            if job.status == JobStatus.COMPLETE:
                to_completed = to_completed.union({job})
            elif job.status == JobStatus.FAILED:
                to_failed = to_failed.union({job})
        self.inprogress = self.inprogress - to_completed - to_failed
        step_completed = len(to_completed)
        self.completed += step_completed
        self.failed += len(to_failed)

        to_inprogress = set()
        to_failed_assigned = set()
        for job in self.assigned:
            if job.status == JobStatus.INPROGRESS:
                to_inprogress = to_inprogress.union({job})
            elif job.status == JobStatus.FAILED:
                to_failed_assigned = to_failed_assigned.union({job})
        self.assigned = self.assigned - to_inprogress - to_failed_assigned
        self.failed += len(to_failed_assigned)
        self.inprogress = self.inprogress.union(to_inprogress)

        to_assigned = set()
        to_rejected = set()
        to_inprogress_arrived = set()
        for job in self.arrived:
            job.tick(self.dt)
            if job.status == JobStatus.ASSIGNED:
                to_assigned = to_assigned.union({job})
            elif job.status == JobStatus.REJECTED:
                to_rejected = to_rejected.union({job})
            elif job.status == JobStatus.INPROGRESS:
                to_inprogress_arrived = to_inprogress_arrived.union({job})
        self.arrived = self.arrived - to_assigned - to_rejected - to_inprogress_arrived
        self.assigned = self.assigned.union(to_assigned)
        self.inprogress = self.inprogress.union(to_inprogress_arrived)
        step_rejected = len(to_rejected)
        self.rejected += step_rejected

        self.t = self.t + datetime.timedelta(seconds=self.dt)
        self.current_price = self.pricing.get_price(self.t)
        self.step_count += 1

        current_capacity = sum([v.battery.actual_capacity for v in self.fleet])
        step_soh_loss_kwh = max(0.0, self.prev_capacity - current_capacity)
        self.prev_capacity = current_capacity

        # --- 4. REWARD CALCULATION (Strictly Peak Shaving + Ride Completion) ---

        V2B_READINESS_REWARD = 0.1  # Hyperparameter to tune
        v2b_readiness_bonus = 0.0

        if peak_active or peak_anticipated:
            for v in self.fleet:
                # If the car is at a charger, has enough battery, and isn't on a job, it's "ready"
                is_at_charger = (v.charger is not None)
                is_available = v.status in [VehicleStatus.IDLE, VehicleStatus.CHARGING, VehicleStatus.DISCHARGING]
                has_enough_soc = v.battery.soc > 0.40 # Or your minimum V2B SOC

                if is_at_charger and is_available and has_enough_soc:
                    v2b_readiness_bonus += V2B_READINESS_REWARD
        
        net_building_load = current_bldg_load + self.current_grid_power
        violation = max(0.0, net_building_load - BUILDING_PEAK_LIMIT)
        
        # Reward only the kW shaved between baseline_load and BUILDING_PEAK_LIMIT (never below 500 kW)
        baseline_violation = max(0.0, current_bldg_load - BUILDING_PEAK_LIMIT)
        peak_shaved_kw = max(0.0, baseline_violation - violation)

        BATTERY_COST_PER_KWH = 135.0
        
        if self.reward_type == "operations":
            # MOBILITY_REWARD = 250.0
            # reward = (step_completed * MOBILITY_REWARD) + (peak_shaved_kw * 20.0)
            # reward -= (step_soh_loss_kwh * BATTERY_COST_PER_KWH)
            # reward -= self.lagrangian_multiplier * violation

            MOBILITY_REWARD = 250.0
            # The new reward structure
            reward = (step_completed * MOBILITY_REWARD)
            reward += (peak_shaved_kw * 20.0)
            reward += v2b_readiness_bonus # <-- ADD THE BONUS HERE
            reward -= (step_soh_loss_kwh * BATTERY_COST_PER_KWH)
            reward -= self.lagrangian_multiplier * violation

        elif self.reward_type == "economics":
            MOBILITY_WEIGHT = 15.0
            fare_revenue = sum([job.fare for job in to_completed])
            reward = (fare_revenue * MOBILITY_WEIGHT) + (peak_shaved_kw * 20.0)
            reward -= (step_soh_loss_kwh * BATTERY_COST_PER_KWH)
            reward -= self.lagrangian_multiplier * violation
        else:
            reward = 0.0
            
        for v in self.fleet:
            if v.status == VehicleStatus.RECOVERY:
                reward -= 1000.0 

        info = {
            'arrived': [j.to_dict() for j in self.arrived],
            'assigned': [j.to_dict() for j in self.assigned],
            'completed': self.completed,
            'step_completed': step_completed,
            'rejected': self.rejected,
            'skipped_rides': step_rejected,
            'inprogress': [j.to_dict() for j in self.inprogress],
            'failed': self.failed,
            'charging_network': [s.to_dict() for s in self.charging_network],
            'fleet': [v.to_dict() for v in self.fleet],
            'total_grid_power': self.current_grid_power,
            'net_building_load': net_building_load,
            'V2B_Revenue': step_v2b_revenue, 
            'current_price': self.current_price,
            'peak_violation': violation 
        }

        return (
            self._get_obs(),
            reward,
            True if self.t >= self.t_max else False,
            False,
            info
        )