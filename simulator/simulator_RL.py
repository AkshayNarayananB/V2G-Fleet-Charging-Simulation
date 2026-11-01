"""Taxi fleet simulator."""

from collections import defaultdict

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
        except (AttributeError, KeyError):
            return 350.0

    def _get_obs(self) -> np.array:
        current_hour = self.t.hour + (self.t.minute / 60.0)
        obs = [item for v in self.fleet for item in (v.battery.actual_capacity / v.battery.initial_capacity, v.battery.soc)]
        obs.extend([
            current_hour,
            self._get_baseline_load(self.t),
            self._get_baseline_load(self.t + datetime.timedelta(minutes=15)),
            self._get_baseline_load(self.t + datetime.timedelta(minutes=30)),
            self._get_baseline_load(self.t + datetime.timedelta(minutes=45))
        ])
        return np.array(obs, dtype=np.float32)

    def get_closest_charger_from_loc(self, loc: Location) -> Tuple[ChargeStation, float]:
        if not self.charging_network: return None, float('inf')
        return min(((charger, d) for charger in self.charging_network if (d := loc.to(charger.location)[0]) is not None), key=lambda x: x[1], default=(self.charging_network[0], float('inf')))

    def get_closest_job(self, vehicle: Vehicle, jobs_by_zone: dict) -> Job:
        """C-optimized O(Z) search using built-in min() instead of manual Python loops."""
        if not jobs_by_zone:
            return None

        veh_zone = vehicle.location.zone
        
        # O(1) early exit: Exact zone match
        if veh_zone in jobs_by_zone and jobs_by_zone[veh_zone]:
            return jobs_by_zone[veh_zone][0]

        region_distances = vehicle.location.region.map.get(veh_zone, {})
        
        # Filter down to zones that actually have jobs AND are reachable from the vehicle
        valid_zones = [z for z, jobs in jobs_by_zone.items() if jobs and z in region_distances]
        
        if not valid_zones:
            return None
            
        # Execute the distance search in C rather than Python bytecode
        best_zone = min(valid_zones, key=lambda z: region_distances[z].get("distance", float('inf')))
        
        if region_distances[best_zone].get("distance", float('inf')) != float('inf'):
            return jobs_by_zone[best_zone][0]
            
        return None

    def _soc_needed_for_distance(self, vehicle: Vehicle, dist_km: float) -> float:
        if dist_km is None or dist_km == float('inf'): return 1.0
        return (dist_km * (vehicle.efficiency / 100.0)) / max(1.0, vehicle.battery.actual_capacity)

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.array, Dict]:
        super().reset(seed=seed)
        self.dt = float(self.config['delta t'])
        self.t = datetime.datetime.strptime(self.config['start t'], '%Y/%m/%d %H:%M:%S')
        self.t_max = datetime.datetime.strptime(self.config['end t'], '%Y/%m/%d %H:%M:%S')
        self.T_a = 25
        self.region = CyclicZoneGraph(self.config['city'])
        self.pricing = PricingUtility(city='nyc', start_date=self.config['start t'].split(' ')[0].replace('/', '-'), end_date=self.config['end t'].split(' ')[0].replace('/', '-'))
        self.current_price = self.pricing.get_price(self.t)
        try:
            self.building_data = pd.read_csv('data/nrel_building_loads.csv', index_col='timestamp', parse_dates=True)
            cluster_mix = self.config.get('microgrid_cluster', {'Office_kW': 2.0})
            self.building_data['Aggregate_kW'] = sum(self.building_data[col] * mult for col, mult in cluster_mix.items() if col in self.building_data)
        except Exception:
            self.building_data = None
            
        self.demand = ReplayDemand(self.config['demand'], self.region)
        self.demand.seek(self.t)
        self.arrived = self.demand.tick(self.dt)
        self.assigned, self.inprogress = set(), set()
        self.rejected, self.completed, self.failed = 0, 0, 0
        
        valid_zones = list(self.region.map.keys())
        spawn_zones = [z for z in [161, 162, 163, 170, 236, 237, 141, 142, 143, 239, 140, 137, 107, 224, 229] if z in valid_zones] or valid_zones
        self.fleet = [Vehicle(model=self.config['fleet']['vehicle'], battery=self.config['fleet']['battery model'], location=CyclicZoneGraphLocation(random.choice(spawn_zones), self.region), vid=i) for i in range(self.config['fleet']['size'])]
        self.charging_network = [ChargeStation(location=CyclicZoneGraphLocation(s['location'], self.region), ports=[ChargePort(s['max port power'], s['efficiency']) for _ in range(s['ports'])], P_max=s['max total power']) for s in self.config['charging stations']]
        
        self.step_count, self.prev_capacity = 0, sum(v.battery.actual_capacity for v in self.fleet)
        
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

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, Dict]:
        # --- PROGRESS TRACKER ---
        if self.step_count % 288 == 0 and self.step_count > 0:
            print(f"[Simulator] Processed Step {self.step_count} (Sim Time: {self.t}) | Active Jobs: {len(self.arrived)}")

        eta = 0.90 
        self.current_grid_power = 0.0
        manual_v2g_power = 0.0 
        step_v2b_revenue = 0.0
        opportunity_cost_penalty = 0.0

        BUILDING_PEAK_LIMIT = 500.0
        current_bldg_load = self._get_baseline_load(self.t)
        peak_active = current_bldg_load > BUILDING_PEAK_LIMIT
        remaining_v2b_needed_kw = max(0.0, current_bldg_load - BUILDING_PEAK_LIMIT)
        
        raw_price = float(self.current_price or 0.20)
        price_per_kwh = (raw_price / 1000.0) if raw_price > 2.0 else max(0.15, raw_price)

        jobs_by_zone = defaultdict(list)
        for j in self.arrived:
            if j.status == JobStatus.ARRIVED:
                jobs_by_zone[j.pickup_location.zone].append(j)

        # for idx, v in enumerate(self.fleet):
        #     if v.status in [VehicleStatus.ONJOB, VehicleStatus.TOPICKUP, VehicleStatus.RECOVERY, VehicleStatus.TOLOC]:
        #         continue

        #     dispatch_intent, c_v_norm = float(action[idx, 0]), float(action[idx, 1])
        #     actual_kw = ((c_v_norm + 0.05) / 0.95) * self.c_max if (c_v_norm < -0.05 and self.enable_v2g) else 0.0
            
        #     target_charger, d_to_charger = self.get_closest_charger_from_loc(v.location)
        #     at_charger = (v.charger is not None) and (v.charger.location == v.location)

        #     # Initialize survival lock if not already set
        #     if not hasattr(v, 'needs_survival_charge'):
        #         v.needs_survival_charge = False

        #     # Lock goes True if battery is critically low
        #     if v.battery.soc < 0.30:
        #         v.needs_survival_charge = True
        #     # Lock goes False once battery is healthy enough, or during an active grid peak
        #     elif v.battery.soc >= 0.85 or peak_active:
        #         v.needs_survival_charge = False

        #     # 1. PEAK SHAVING OVERRIDE (Top Priority when grid is stressed & battery > 45%)
        #     building_needs_v2g = self.enable_v2g and peak_active and remaining_v2b_needed_kw > 0.1
            
        #     if building_needs_v2g and v.battery.soc > 0.45:
        #         # Do not interrupt survival mode for V2B if battery is critically low
        #         if v.battery.soc > 0.35:
        #             if not at_charger:
        #                 v.charge(target_charger, -self.c_max)
        #                 continue
                    
        #             allocated_kw = min(self.c_max, remaining_v2b_needed_kw)
        #             if v.charger and v.charger != target_charger: v.charger.disconnect(v.vid)
        #             v.charger = target_charger
        #             v.preferred_rate = -allocated_kw
        #             v.status = VehicleStatus.DISCHARGING
                    
        #             max_removable_kwh = max(0.0, (v.battery.soc - 0.40) * v.battery.actual_capacity)
        #             desired_kwh = (allocated_kw / eta) * (self.dt / 3600.0)
        #             actual_kwh = min(max_removable_kwh, desired_kwh)
                    
        #             if actual_kwh > 0.01:
        #                 v.battery.discharge(actual_kwh, self.dt, self.T_a)
        #                 delivered_kwh = actual_kwh * eta
        #                 effective_kw = delivered_kwh / (self.dt / 3600.0)
        #                 manual_v2g_power -= effective_kw
        #                 remaining_v2b_needed_kw = max(0.0, remaining_v2b_needed_kw - effective_kw)
        #                 step_v2b_revenue += delivered_kwh * max(0.25, price_per_kwh * 2.0)
        #             continue

        #     # 2. RIDE DISPATCH (Must have 35% battery and not be locked in recovery)
        #     closest_job = self.get_closest_job(v, jobs_by_zone)
        #     if closest_job and not v.needs_survival_charge:
        #         d_pickup, _ = v.location.to(closest_job.pickup_location)
        #         d_ride, _ = closest_job.pickup_location.to(closest_job.dropoff_location)
        #         _, d_drop_chg = self.get_closest_charger_from_loc(closest_job.dropoff_location)
        #         req_soc = self._soc_needed_for_distance(v, d_pickup + d_ride + d_drop_chg) + 0.15 if (d_pickup and d_ride) else 1.0
                
        #         if v.battery.soc >= max(0.35, req_soc) and (dispatch_intent >= -0.2 or v.battery.soc > 0.60):
        #             if v.charger: v.charger.disconnect(v.vid)
        #             v.service_demand(closest_job)
        #             self.arrived.remove(closest_job)
        #             self.assigned.add(closest_job)
        #             jobs_by_zone[closest_job.pickup_location.zone].remove(closest_job)
        #             if not jobs_by_zone[closest_job.pickup_location.zone]: 
        #                 del jobs_by_zone[closest_job.pickup_location.zone]
        #             continue

        #     # 3. CRITICAL SURVIVAL / RECOVERY CHARGING (Agent cannot interrupt this)
        #     if v.needs_survival_charge:
        #         v.charge(target_charger, self.c_max)
        #         if at_charger: v.status = VehicleStatus.CHARGING
        #         continue

        #     # 4. AGENT CHARGING / DEFAULT IDLE (Only during off-peak)
        #     if actual_kw > 2.0 and v.battery.soc < 0.90:
        #         if peak_active:
        #             opportunity_cost_penalty += 50.0
        #             if v.charger: v.charger.disconnect(v.vid)
        #             v.status = VehicleStatus.IDLE
        #         else:
        #             v.charge(target_charger, actual_kw)
        #             if at_charger: v.status = VehicleStatus.CHARGING
        #         continue

        #     if v.charger: v.charger.disconnect(v.vid)
        #     v.status = VehicleStatus.IDLE

        for idx, v in enumerate(self.fleet):
            if v.status in [VehicleStatus.ONJOB, VehicleStatus.TOPICKUP, VehicleStatus.RECOVERY, VehicleStatus.TOLOC]:
                continue

            dispatch_intent, c_v_norm = float(action[idx, 0]), float(action[idx, 1])
            actual_kw = ((c_v_norm + 0.05) / 0.95) * self.c_max if (c_v_norm < -0.05 and self.enable_v2g) else 0.0
            if c_v_norm > 0.05: actual_kw = ((c_v_norm - 0.05) / 0.95) * self.c_max
            
            target_charger, d_to_charger = self.get_closest_charger_from_loc(v.location)
            at_charger = (v.charger is not None) and (v.charger.location == v.location)

            # 1. RIDE DISPATCH (Top Priority - Maximize Mobility Revenue)
            closest_job = self.get_closest_job(v, jobs_by_zone)
            if closest_job:
                d_pickup, _ = v.location.to(closest_job.pickup_location)
                d_ride, _ = closest_job.pickup_location.to(closest_job.dropoff_location)
                _, d_drop_chg = self.get_closest_charger_from_loc(closest_job.dropoff_location)
                req_soc = self._soc_needed_for_distance(v, d_pickup + d_ride + d_drop_chg) + 0.15 if (d_pickup and d_ride) else 1.0
                
                if v.battery.soc >= max(0.35, req_soc):
                    if v.charger: v.charger.disconnect(v.vid)
                    v.service_demand(closest_job)
                    self.arrived.remove(closest_job)
                    self.assigned.add(closest_job)
                    jobs_by_zone[closest_job.pickup_location.zone].remove(closest_job)
                    if not jobs_by_zone[closest_job.pickup_location.zone]: 
                        del jobs_by_zone[closest_job.pickup_location.zone]
                    continue

            # 2. CRITICAL SURVIVAL (Prevent dead vehicles on the road)
            if v.battery.soc < 0.30:
                v.charge(target_charger, self.c_max)
                if at_charger: v.status = VehicleStatus.CHARGING
                continue

            # 3. DEFAULT FLEET ROUTING (Unused cars physically relocate to chargers)
            if not at_charger:
                v.charge(target_charger, self.c_max)
                continue

            # 4. PPO AGENT DIRECT CONTROL (Cars are plugged in and awaiting commands)
            if actual_kw < -2.0 and v.battery.soc > 0.40 and self.enable_v2g:
                # Agent chooses V2B Discharging
                allocated_kw = abs(actual_kw)
                if v.charger and v.charger != target_charger: v.charger.disconnect(v.vid)
                v.charger = target_charger
                v.preferred_rate = -allocated_kw
                v.status = VehicleStatus.DISCHARGING
                
                # Execute manual V2B physics
                max_removable_kwh = max(0.0, (v.battery.soc - 0.40) * v.battery.actual_capacity)
                desired_kwh = (allocated_kw / eta) * (self.dt / 3600.0)
                actual_kwh = min(max_removable_kwh, desired_kwh)
                
                if actual_kwh > 0.01:
                    v.battery.discharge(actual_kwh, self.dt, self.T_a)
                    delivered_kwh = actual_kwh * eta
                    effective_kw = delivered_kwh / (self.dt / 3600.0)
                    manual_v2g_power -= effective_kw
                    step_v2b_revenue += delivered_kwh * max(0.25, price_per_kwh * 2.0)
                continue
                
            elif actual_kw > 2.0 and v.battery.soc < 0.95:
                # Agent chooses Charging
                if peak_active:
                    opportunity_cost_penalty += 50.0  # Penalize agent for worsening the peak
                v.charge(target_charger, actual_kw)
                v.status = VehicleStatus.CHARGING
                continue

            # 5. AGENT CHOOSES IDLE (0.0 kW)
            if v.charger: v.charger.disconnect(v.vid)
            v.status = VehicleStatus.IDLE

        for vehicle in self.fleet: vehicle.tick(self.dt, {'T_a': self.T_a})
        
        for charger in self.charging_network:
            charger.tick(self.fleet, self.dt, self.T_a)
            for port in charger.ports:
                if port.vehicle and port.P_t > 0:
                    self.current_grid_power += port.P_t
        
        # Apply the manual V2G offset to the total grid power
        self.current_grid_power += manual_v2g_power
        
        self.arrived.update(self.demand.tick(self.dt))
        
        to_completed = {j for j in self.inprogress if j.status == JobStatus.COMPLETE}
        to_failed = {j for j in self.inprogress if j.status == JobStatus.FAILED}
        self.inprogress -= (to_completed | to_failed)
        self.completed += len(to_completed)
        self.failed += len(to_failed)

        to_inprogress = {j for j in self.assigned if j.status == JobStatus.INPROGRESS}
        to_failed_assigned = {j for j in self.assigned if j.status == JobStatus.FAILED}
        self.assigned -= (to_inprogress | to_failed_assigned)
        self.inprogress.update(to_inprogress)
        self.failed += len(to_failed_assigned)

        to_assigned = set()
        to_rejected = set()
        to_inprogress_arrived = set()
        
        # CRITICAL FIX: Tick jobs so they expire properly, stopping the infinite queue growth
        for job in self.arrived:
            job.tick(self.dt)
            if job.status == JobStatus.ASSIGNED:
                to_assigned.add(job)
            elif job.status == JobStatus.REJECTED:
                to_rejected.add(job)
            elif job.status == JobStatus.INPROGRESS:
                to_inprogress_arrived.add(job)
                
        self.arrived -= (to_assigned | to_rejected | to_inprogress_arrived)
        self.assigned.update(to_assigned)
        self.inprogress.update(to_inprogress_arrived)
        step_rejected = len(to_rejected)
        self.rejected += step_rejected
        
        self.t += datetime.timedelta(seconds=self.dt)
        self.current_price = self.pricing.get_price(self.t)
        self.step_count += 1
        current_capacity = sum([v.battery.actual_capacity for v in self.fleet])
        step_soh_loss_kwh = max(0.0, self.prev_capacity - current_capacity)
        self.prev_capacity = current_capacity
        
        net_building_load = current_bldg_load + self.current_grid_power
        violation = max(0.0, net_building_load - BUILDING_PEAK_LIMIT)
        baseline_violation = max(0.0, current_bldg_load - BUILDING_PEAK_LIMIT)
        peak_shaved_kw = max(0.0, baseline_violation - violation)
        BATTERY_COST_PER_KWH = 135.0
        
        reward = 0.0
        if self.reward_type == "operations":
            MOBILITY_REWARD = 250.0
            V2B_REWARD_MULTIPLIER = 50.0
            reward += (len(to_completed) * MOBILITY_REWARD)
            reward += (peak_shaved_kw * V2B_REWARD_MULTIPLIER)
            reward -= (step_soh_loss_kwh * BATTERY_COST_PER_KWH)
            reward -= (self.lagrangian_multiplier * violation)
            reward -= opportunity_cost_penalty
        elif self.reward_type == "economics":
            MOBILITY_WEIGHT = 15.0
            fare_revenue = sum([job.fare for job in to_completed])
            reward += (fare_revenue * MOBILITY_WEIGHT)
            reward += (peak_shaved_kw * 20.0) 
            reward -= (step_soh_loss_kwh * BATTERY_COST_PER_KWH)
            reward -= (self.lagrangian_multiplier * violation)
            reward -= opportunity_cost_penalty

        for v in self.fleet:
            if v.status == VehicleStatus.RECOVERY:
                reward -= 1000.0

        class FastInfo(dict):
            def __deepcopy__(self, memo):
                return self

        info = FastInfo({
            'assigned': [j.to_dict() for j in self.assigned],
            'completed': self.completed,
            'step_completed': len(to_completed),
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
        })
        
        return self._get_obs(), reward, self.t >= self.t_max, False, info