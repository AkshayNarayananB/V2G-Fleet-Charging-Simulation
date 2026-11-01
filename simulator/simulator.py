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
    """Taxi fleet simulator with Hard Action Projection for V2B Peak Shaving."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config

        self.reward_type = self.config.get('reward_type', 'economics').lower()
        self.enable_v2g = self.config.get('enable_v2g', True)

        fleet_size = self.config['fleet']['size']
        self.c_max = (
            self.config['charging stations'][0]['max port power']
            if self.config.get('charging stations')
            else 50.0
        )
        obs_dim = (fleet_size * 2) + 5
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        low_action = np.full((fleet_size, 2), -1.0, dtype=np.float32)
        high_action = np.full((fleet_size, 2), 1.0, dtype=np.float32)
        if not self.enable_v2g:
            low_action[:, 1] = 0.0

        self.action_space = gym.spaces.Box(
            low=low_action, high=high_action, shape=(fleet_size, 2), dtype=np.float32
        )

    def _get_baseline_load(self, timestamp: datetime.datetime) -> float:
        minute_floor = (timestamp.minute // 15) * 15
        lookup_time = timestamp.replace(minute=minute_floor, second=0, microsecond=0)
        try:
            return float(self.building_data.loc[lookup_time, 'Aggregate_kW'])
        except (AttributeError, KeyError):
            return 350.0

    def _get_obs(self) -> np.array:
        current_hour = self.t.hour + (self.t.minute / 60.0)
        obs = [
            item
            for v in self.fleet
            for item in (
                v.battery.actual_capacity / v.battery.initial_capacity,
                v.battery.soc,
            )
        ]
        obs.extend([
            current_hour,
            self._get_baseline_load(self.t),
            self._get_baseline_load(self.t + datetime.timedelta(minutes=15)),
            self._get_baseline_load(self.t + datetime.timedelta(minutes=30)),
            self._get_baseline_load(self.t + datetime.timedelta(minutes=45)),
        ])
        return np.array(obs, dtype=np.float32)

    def _dist_between(self, loc1: Location, loc2: Location) -> float:
        if loc1.zone == loc2.zone:
            return 0.0
        d, _ = loc1.to(loc2)
        return float(d) if d is not None else float('inf')

    def get_closest_charger_from_loc(self, loc: Location) -> Tuple[ChargeStation, float]:
        if not self.charging_network:
            return None, float('inf')
        return min(
            ((charger, self._dist_between(loc, charger.location)) for charger in self.charging_network),
            key=lambda x: x[1],
            default=(self.charging_network[0], float('inf')),
        )

    def get_best_charger_from_loc(self, loc: Location) -> Tuple[ChargeStation, float]:
        """Routes vehicles to the closest charger that currently has an open port."""
        if not self.charging_network:
            return None, float('inf')
        free_chargers = [
            (charger, self._dist_between(loc, charger.location))
            for charger in self.charging_network
            if any(p.vehicle is None for p in charger.ports)
        ]
        if free_chargers:
            best = min(free_chargers, key=lambda x: x[1])
            if best[1] != float('inf'):
                return best
        return self.get_closest_charger_from_loc(loc)

    def get_closest_job(self, vehicle: Vehicle, jobs_by_zone: dict) -> Job:
        """C-optimized O(Z) search using built-in min() instead of manual Python loops."""
        if not jobs_by_zone:
            return None

        veh_zone = vehicle.location.zone

        if veh_zone in jobs_by_zone and jobs_by_zone[veh_zone]:
            return jobs_by_zone[veh_zone][0]

        region_distances = vehicle.location.region.map.get(veh_zone, {})
        valid_zones = [
            z for z, jobs in jobs_by_zone.items() if jobs and z in region_distances
        ]

        if not valid_zones:
            return None

        best_zone = min(
            valid_zones,
            key=lambda z: region_distances[z].get("distance", float('inf')),
        )

        if region_distances[best_zone].get("distance", float('inf')) != float('inf'):
            return jobs_by_zone[best_zone][0]

        return None

    def _soc_needed_for_distance(self, vehicle: Vehicle, dist_km: float) -> float:
        if dist_km is None or dist_km == float('inf'):
            return 1.0
        return (dist_km * (vehicle.efficiency / 100.0)) / max(1.0, vehicle.battery.actual_capacity)

    @staticmethod
    def _is_vehicle_on_charger(vehicle: Vehicle, charger: ChargeStation) -> bool:
        if charger is None:
            return False
        return any(
            (p.vehicle == vehicle.vid or p.vehicle is vehicle) for p in charger.ports
        )

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.array, Dict]:
        super().reset(seed=seed)
        self.dt = float(self.config['delta t'])
        self.t = datetime.datetime.strptime(self.config['start t'], '%Y/%m/%d %H:%M:%S')
        self.t_max = datetime.datetime.strptime(self.config['end t'], '%Y/%m/%d %H:%M:%S')
        self.T_a = 25
        self.region = CyclicZoneGraph(self.config['city'])
        self.pricing = PricingUtility(
            city='nyc',
            start_date=self.config['start t'].split(' ')[0].replace('/', '-'),
            end_date=self.config['end t'].split(' ')[0].replace('/', '-'),
        )
        self.current_price = self.pricing.get_price(self.t)
        try:
            self.building_data = pd.read_csv(
                'data/nrel_building_loads.csv', index_col='timestamp', parse_dates=True
            )
            cluster_mix = self.config.get('microgrid_cluster', {'Office_kW': 2.0})
            self.building_data['Aggregate_kW'] = sum(
                self.building_data[col] * mult
                for col, mult in cluster_mix.items()
                if col in self.building_data
            )
        except Exception:
            self.building_data = None

        self.demand = ReplayDemand(self.config['demand'], self.region)
        self.demand.seek(self.t)
        self.arrived = self.demand.tick(self.dt)
        self.assigned, self.inprogress = set(), set()
        self.rejected, self.completed, self.failed = 0, 0, 0

        valid_zones = list(self.region.map.keys())
        spawn_zones = [
            z
            for z in [161, 162, 163, 170, 236, 237, 141, 142, 143, 239, 140, 137, 107, 224, 229]
            if z in valid_zones
        ] or valid_zones

        self.charging_network = [
            ChargeStation(
                location=CyclicZoneGraphLocation(s['location'], self.region),
                ports=[ChargePort(s['max port power'], s['efficiency']) for _ in range(s['ports'])],
                P_max=s['max total power'],
            )
            for s in self.config['charging stations']
        ]
        charger_zones = [c.location.zone for c in self.charging_network if c.location.zone in valid_zones]
        initial_zones = charger_zones if charger_zones else spawn_zones

        self.fleet = [
            Vehicle(
                model=self.config['fleet']['vehicle'],
                battery=self.config['fleet']['battery model'],
                location=CyclicZoneGraphLocation(random.choice(initial_zones), self.region),
                vid=i,
            )
            for i in range(self.config['fleet']['size'])
        ]

        self.step_count = 0
        self.prev_capacity = sum(v.battery.actual_capacity for v in self.fleet)

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
            'net_building_load': self._get_baseline_load(self.t),
        }
        return self._get_obs(), info

    def _discharge_vehicle_v2b(
        self, v: Vehicle, target_charger: ChargeStation, allocated_kw: float, eta: float, price_per_kwh: float
    ) -> Tuple[float, float]:
        """Executes V2B discharging with guaranteed battery energy conservation."""
        if v.charger and v.charger != target_charger:
            v.charger.disconnect(v.vid)
        if self._is_vehicle_on_charger(v, target_charger):
            target_charger.disconnect(v.vid)

        v.charger = target_charger
        v.preferred_rate = -allocated_kw
        v.status = VehicleStatus.DISCHARGING

        max_removable_kwh = max(0.0, (v.battery.soc - 0.32) * v.battery.actual_capacity)
        desired_kwh = (allocated_kw / eta) * (self.dt / 3600.0)
        actual_kwh = min(max_removable_kwh, desired_kwh)

        if actual_kwh <= 0.005:
            v.status = VehicleStatus.IDLE
            return 0.0, 0.0

        soc_before = v.battery.soc
        v.battery.discharge(actual_kwh, self.dt, self.T_a)
        removed_kwh = max(0.0, (soc_before - v.battery.soc) * v.battery.actual_capacity)

        # If battery.discharge internally treated arg1 as kW and multiplied by dt/3600,
        # enforce exact energy deduction so SoC reflects real V2B energy export.
        if removed_kwh < actual_kwh * 0.5 and v.battery.actual_capacity > 0:
            v.battery.soc = max(0.30, soc_before - (actual_kwh / v.battery.actual_capacity))

        delivered_kwh = actual_kwh * eta
        effective_kw = delivered_kwh / (self.dt / 3600.0)
        revenue = delivered_kwh * max(0.25, price_per_kwh * 2.0)
        return effective_kw, revenue

    # def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, Dict]:
    #     if self.step_count % 288 == 0 and self.step_count > 0:
    #         mean_soc = np.mean([v.battery.soc for v in self.fleet]) * 100.0
    #         print(
    #             f"[Simulator] Step {self.step_count} ({self.t}) | "
    #             f"Completed Rides: {self.completed} | Mean SoC: {mean_soc:.1f}%"
    #         )

    #     eta = 0.90
    #     self.current_grid_power = 0.0
    #     manual_v2g_power = 0.0
    #     step_v2b_revenue = 0.0
    #     agent_alignment_bonus = 0.0

    #     BUILDING_PEAK_LIMIT = 500.0
    #     current_bldg_load = self._get_baseline_load(self.t)
    #     peak_active = current_bldg_load > BUILDING_PEAK_LIMIT

    #     raw_price = float(self.current_price or 0.20)
    #     price_per_kwh = (raw_price / 1000.0) if raw_price > 2.0 else max(0.15, raw_price)

    #     jobs_by_zone = defaultdict(list)
    #     for j in self.arrived:
    #         if j.status == JobStatus.ARRIVED:
    #             jobs_by_zone[j.pickup_location.zone].append(j)

    #     chargers_by_zone = defaultdict(list)
    #     for c in self.charging_network:
    #         chargers_by_zone[c.location.zone].append(c)

    #     available_for_grid = []

    #     # =====================================================================
    #     # PHASE 1: MOBILITY DISPATCH & CHARGER ROUTING
    #     # =====================================================================
    #     for idx, v in enumerate(self.fleet):
    #         if v.status in [
    #             VehicleStatus.ONJOB,
    #             VehicleStatus.TOPICKUP,
    #             VehicleStatus.RECOVERY,
    #             VehicleStatus.TOLOC,
    #         ]:
    #             continue

    #         c_v_norm = float(action[idx, 1]) if action is not None else 0.0

    #         # 1. RIDE DISPATCH (Top Priority - Never strand rides if SoC >= 35%)
    #         closest_job = self.get_closest_job(v, jobs_by_zone)
    #         if closest_job:
    #             d_pickup = self._dist_between(v.location, closest_job.pickup_location)
    #             d_ride = self._dist_between(closest_job.pickup_location, closest_job.dropoff_location)
    #             _, d_drop_chg = self.get_closest_charger_from_loc(closest_job.dropoff_location)
    #             req_soc = (
    #                 self._soc_needed_for_distance(v, d_pickup + d_ride + d_drop_chg) + 0.12
    #                 if (d_pickup != float('inf') and d_ride != float('inf'))
    #                 else 0.35
    #             )

    #             if v.battery.soc >= max(0.35, min(0.65, req_soc)):
    #                 if v.charger:
    #                     v.charger.disconnect(v.vid)
    #                 v.service_demand(closest_job)
    #                 self.arrived.remove(closest_job)
    #                 self.assigned.add(closest_job)
    #                 jobs_by_zone[closest_job.pickup_location.zone].remove(closest_job)
    #                 if not jobs_by_zone[closest_job.pickup_location.zone]:
    #                     del jobs_by_zone[closest_job.pickup_location.zone]
    #                 continue

    #         # 2. IN-TRANSIT GUARD (Never reset route while vehicle is driving to charger)
    #         if v.status == VehicleStatus.TOCHARGE and v.location.zone not in chargers_by_zone:
    #             v.preferred_rate = 0.0 if peak_active else self.c_max
    #             continue

    #         # 3. ROUTE IDLE VEHICLES IN NON-CHARGER ZONES TO AN OPEN CHARGER
    #         if v.location.zone not in chargers_by_zone:
    #             best_charger, _ = self.get_best_charger_from_loc(v.location)
    #             if best_charger is not None:
    #                 v.charge(best_charger, 0.0 if peak_active else self.c_max)
    #             continue

    #         # 4. VEHICLE IS PHYSICALLY AT A CHARGER ZONE
    #         zone_chargers = chargers_by_zone[v.location.zone]
    #         target_charger = v.charger if (v.charger in zone_chargers) else zone_chargers[0]
    #         for c in zone_chargers:
    #             if self._is_vehicle_on_charger(v, c):
    #                 target_charger = c
    #                 break
    #             if any(p.vehicle is None for p in c.ports):
    #                 target_charger = c

    #         available_for_grid.append((idx, v, target_charger, c_v_norm))

    #     # =====================================================================
    #     # PHASE 2: ACTION PROJECTION (HARD 500 kW CEILING)
    #     # =====================================================================
    #     if peak_active:
    #         # --- REGIME A: BUILDING PEAK ACTIVE (> 500 kW) ---
    #         # 1. Zero charging allowed: disconnect any vehicle occupying a charging port
    #         for _, v, tc, c_v_norm in available_for_grid:
    #             if v.battery.soc > 0.40 and c_v_norm < -0.05:
    #                 agent_alignment_bonus += 0.01
    #             if v.charger:
    #                 v.charger.disconnect(v.vid)
    #             v.status = VehicleStatus.IDLE

    #         # 2. Rank capable V2B vehicles by agent discharge intent and SoC
    #         remaining_v2b_kw = current_bldg_load - BUILDING_PEAK_LIMIT
    #         v2b_candidates = [
    #             item for item in available_for_grid if item[1].battery.soc > 0.35 and self.enable_v2g
    #         ]
    #         v2b_candidates.sort(key=lambda x: (x[3], -x[1].battery.soc))

    #         for _, v, tc, _ in v2b_candidates:
    #             if remaining_v2b_kw <= 0.05:
    #                 break
    #             alloc_kw = min(self.c_max, remaining_v2b_kw)
    #             eff_kw, rev = self._discharge_vehicle_v2b(v, tc, alloc_kw, eta, price_per_kwh)
    #             manual_v2g_power -= eff_kw
    #             remaining_v2b_kw = max(0.0, remaining_v2b_kw - eff_kw)
    #             step_v2b_revenue += rev

    #     else:
    #         # --- REGIME B: OFF-PEAK SAFE HEADROOM (<= 500 kW) ---
    #         # Leave a 10 kW safety buffer below 500 kW so charging never creates a new peak
    #         remaining_charge_headroom_kw = max(0.0, (BUILDING_PEAK_LIMIT - 10.0) - current_bldg_load)

    #         # Prioritize lowest-SoC vehicles first so no vehicle ever gets stranded at 33%
    #         available_for_grid.sort(key=lambda x: x[1].battery.soc)

    #         for _, v, tc, c_v_norm in available_for_grid:
    #             if v.battery.soc < 0.75 and c_v_norm > 0.05:
    #                 agent_alignment_bonus += 0.01

    #             # Vehicles >= 85% SoC disconnect and yield physical ports to depleted cars
    #             if v.battery.soc >= 0.85:
    #                 if v.charger:
    #                     v.charger.disconnect(v.vid)
    #                 v.status = VehicleStatus.IDLE
    #                 continue

    #             # Determine desired charge power (agent modulates above 55% SoC)
    #             if v.battery.soc < 0.55:
    #                 desired_kw = self.c_max
    #             else:
    #                 if c_v_norm < -0.35:
    #                     desired_kw = 0.0
    #                 else:
    #                     agent_frac = np.clip((c_v_norm + 1.0) / 2.0, 0.35, 1.0)
    #                     desired_kw = agent_frac * self.c_max

    #             if desired_kw > 1.0 and remaining_charge_headroom_kw >= self.c_max:
    #                 final_kw = min(self.c_max, desired_kw)
    #                 is_plugged = self._is_vehicle_on_charger(v, tc)
    #                 if is_plugged:
    #                     v.preferred_rate = final_kw
    #                     v.status = VehicleStatus.CHARGING
    #                 elif v.status == VehicleStatus.TOCHARGE and v.charger == tc:
    #                     v.preferred_rate = final_kw
    #                 else:
    #                     v.charge(tc, final_kw)
    #                 # Budget full port power so charger.tick() can never exceed headroom
    #                 remaining_charge_headroom_kw -= self.c_max
    #             else:
    #                 if v.charger:
    #                     v.charger.disconnect(v.vid)
    #                 v.status = VehicleStatus.IDLE

    #     # =====================================================================
    #     # PHASE 3: PHYSICS TICK & POST-TICK PEAK GUARANTEE
    #     # =====================================================================
    #     for vehicle in self.fleet:
    #         vehicle.tick(self.dt, {'T_a': self.T_a})

    #     # Safety gate: if a TOCHARGE vehicle arrived during vehicle.tick() while a peak is active,
    #     # disconnect its port before charger.tick() so it cannot draw grid power during a peak.
    #     if peak_active:
    #         for charger in self.charging_network:
    #             for port in charger.ports:
    #                 if port.vehicle is not None:
    #                     vid = port.vehicle if isinstance(port.vehicle, int) else port.vehicle.vid
    #                     charger.disconnect(vid)

    #     for charger in self.charging_network:
    #         charger.tick(self.fleet, self.dt, self.T_a)
    #         for port in charger.ports:
    #             if port.vehicle and port.P_t > 0:
    #                 self.current_grid_power += port.P_t

    #     # Secondary Projection Guarantee: if net load (building + charging + V2B) still > 500 kW,
    #     # immediately dispatch reserve idle vehicles at charger zones to shave the residual excess.
    #     prelim_net_load = current_bldg_load + self.current_grid_power + manual_v2g_power
    #     if prelim_net_load > BUILDING_PEAK_LIMIT and self.enable_v2g:
    #         residual_kw = prelim_net_load - BUILDING_PEAK_LIMIT
    #         reserve_v2b = [
    #             (v, tc)
    #             for _, v, tc, _ in available_for_grid
    #             if v.status == VehicleStatus.IDLE and v.battery.soc > 0.35
    #         ]
    #         reserve_v2b.sort(key=lambda x: -x[0].battery.soc)
    #         for v, tc in reserve_v2b:
    #             if residual_kw <= 0.05:
    #                 break
    #             alloc_kw = min(self.c_max, residual_kw)
    #             eff_kw, rev = self._discharge_vehicle_v2b(v, tc, alloc_kw, eta, price_per_kwh)
    #             manual_v2g_power -= eff_kw
    #             residual_kw = max(0.0, residual_kw - eff_kw)
    #             step_v2b_revenue += rev

    #     self.current_grid_power += manual_v2g_power

    #     # =====================================================================
    #     # PHASE 4: DEMAND QUEUE & REWARD COMPUTATION
    #     # =====================================================================
    #     self.arrived.update(self.demand.tick(self.dt))

    #     to_completed = {j for j in self.inprogress if j.status == JobStatus.COMPLETE}
    #     to_failed = {j for j in self.inprogress if j.status == JobStatus.FAILED}
    #     self.inprogress -= (to_completed | to_failed)
    #     self.completed += len(to_completed)
    #     self.failed += len(to_failed)

    #     to_inprogress = {j for j in self.assigned if j.status == JobStatus.INPROGRESS}
    #     to_failed_assigned = {j for j in self.assigned if j.status == JobStatus.FAILED}
    #     self.assigned -= (to_inprogress | to_failed_assigned)
    #     self.inprogress.update(to_inprogress)
    #     self.failed += len(to_failed_assigned)

    #     to_assigned = set()
    #     to_rejected = set()
    #     to_inprogress_arrived = set()

    #     for job in self.arrived:
    #         job.tick(self.dt)
    #         if job.status == JobStatus.ASSIGNED:
    #             to_assigned.add(job)
    #         elif job.status == JobStatus.REJECTED:
    #             to_rejected.add(job)
    #         elif job.status == JobStatus.INPROGRESS:
    #             to_inprogress_arrived.add(job)

    #     self.arrived -= (to_assigned | to_rejected | to_inprogress_arrived)
    #     self.assigned.update(to_assigned)
    #     self.inprogress.update(to_inprogress_arrived)
    #     step_rejected = len(to_rejected)
    #     self.rejected += step_rejected

    #     self.t += datetime.timedelta(seconds=self.dt)
    #     self.current_price = self.pricing.get_price(self.t)
    #     self.step_count += 1
    #     current_capacity = sum(v.battery.actual_capacity for v in self.fleet)
    #     step_soh_loss_kwh = max(0.0, self.prev_capacity - current_capacity)
    #     self.prev_capacity = current_capacity

    #     net_building_load = current_bldg_load + self.current_grid_power
    #     violation = max(0.0, net_building_load - BUILDING_PEAK_LIMIT)
    #     baseline_violation = max(0.0, current_bldg_load - BUILDING_PEAK_LIMIT)
    #     peak_shaved_kw = max(0.0, baseline_violation - violation)

    #     # Normalized O(1) reward structure focused on ride completion & peak shaving
    #     reward = 0.0
    #     reward += 2.0 * len(to_completed)
    #     reward += 1.5 * (peak_shaved_kw / 100.0)
    #     if peak_active and violation < 1.0:
    #         reward += 1.0
    #     reward += agent_alignment_bonus
    #     reward -= 0.5 * step_soh_loss_kwh
    #     reward -= 2.0 * (violation / 100.0)

    #     class FastInfo(dict):
    #         def __deepcopy__(self, memo):
    #             return self

    #     info = FastInfo({
    #         'arrived': [j.to_dict() for j in self.arrived],
    #         'assigned': [j.to_dict() for j in self.assigned],
    #         'completed': self.completed,
    #         'step_completed': len(to_completed),
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
    #         'peak_violation': violation,
    #     })

    #     return self._get_obs(), float(reward), self.t >= self.t_max, False, info

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, Dict]:
        if self.step_count % 288 == 0 and self.step_count > 0:
            mean_soc = np.mean([v.battery.soc for v in self.fleet]) * 100.0
            print(
                f"[Simulator] Step {self.step_count} ({self.t}) | "
                f"Completed Rides: {self.completed} | Mean SoC: {mean_soc:.1f}%"
            )

        eta = 0.90
        self.current_grid_power = 0.0
        manual_v2g_power = 0.0
        step_v2b_revenue = 0.0
        agent_alignment_bonus = 0.0

        BUILDING_PEAK_LIMIT = 500.0
        current_bldg_load = self._get_baseline_load(self.t)
        peak_active = current_bldg_load > BUILDING_PEAK_LIMIT

        raw_price = float(self.current_price or 0.20)
        price_per_kwh = (raw_price / 1000.0) if raw_price > 2.0 else max(0.15, raw_price)

        jobs_by_zone = defaultdict(list)
        for j in self.arrived:
            if j.status == JobStatus.ARRIVED:
                jobs_by_zone[j.pickup_location.zone].append(j)

        chargers_by_zone = defaultdict(list)
        for c in self.charging_network:
            chargers_by_zone[c.location.zone].append(c)

        # =====================================================================
        # PHASE 0: THE V2B RESERVE LOCK
        # =====================================================================
        reserved_vids_for_grid = set()
        if peak_active and self.enable_v2g:
            deficit = current_bldg_load - BUILDING_PEAK_LIMIT
            cars_needed = int(np.ceil(deficit / self.c_max))
            
            # Find vehicles physically at a charger with enough juice to help
            potential_reserves = [
                v for v in self.fleet 
                if v.status not in [VehicleStatus.ONJOB, VehicleStatus.TOPICKUP] 
                and v.location.zone in chargers_by_zone
                and v.battery.soc > 0.35
            ]
            # Sort by highest SoC to use the most charged batteries first
            potential_reserves.sort(key=lambda x: -x.battery.soc)
            
            # Lock the exact number of cars needed to save the grid
            for v in potential_reserves[:cars_needed]:
                reserved_vids_for_grid.add(v.vid)

        available_for_grid = []

        # =====================================================================
        # PHASE 1: MOBILITY DISPATCH & CHARGER ROUTING
        # =====================================================================
        for idx, v in enumerate(self.fleet):
            if v.status in [
                VehicleStatus.ONJOB,
                VehicleStatus.TOPICKUP,
                VehicleStatus.RECOVERY,
                VehicleStatus.TOLOC,
            ]:
                continue

            c_v_norm = float(action[idx, 1]) if action is not None else 0.0

            # 1. RIDE DISPATCH (Bypass cars that are locked for grid survival)
            if v.vid not in reserved_vids_for_grid:
                closest_job = self.get_closest_job(v, jobs_by_zone)
                if closest_job:
                    d_pickup = self._dist_between(v.location, closest_job.pickup_location)
                    d_ride = self._dist_between(closest_job.pickup_location, closest_job.dropoff_location)
                    _, d_drop_chg = self.get_closest_charger_from_loc(closest_job.dropoff_location)
                    req_soc = (
                        self._soc_needed_for_distance(v, d_pickup + d_ride + d_drop_chg) + 0.12
                        if (d_pickup != float('inf') and d_ride != float('inf'))
                        else 0.35
                    )

                    if v.battery.soc >= max(0.35, min(0.65, req_soc)):
                        if v.charger:
                            v.charger.disconnect(v.vid)
                        v.service_demand(closest_job)
                        self.arrived.remove(closest_job)
                        self.assigned.add(closest_job)
                        jobs_by_zone[closest_job.pickup_location.zone].remove(closest_job)
                        if not jobs_by_zone[closest_job.pickup_location.zone]:
                            del jobs_by_zone[closest_job.pickup_location.zone]
                        continue

            # 2. IN-TRANSIT GUARD
            if v.status == VehicleStatus.TOCHARGE and v.location.zone not in chargers_by_zone:
                v.preferred_rate = 0.0 if peak_active else self.c_max
                continue

            # 3. ROUTE IDLE VEHICLES
            if v.location.zone not in chargers_by_zone:
                best_charger, _ = self.get_best_charger_from_loc(v.location)
                if best_charger is not None:
                    v.charge(best_charger, 0.0 if peak_active else self.c_max)
                continue
            

            # 4. VEHICLE IS PHYSICALLY AT A CHARGER ZONE
            zone_chargers = chargers_by_zone[v.location.zone]
            target_charger = v.charger if (v.charger in zone_chargers) else zone_chargers[0]
            for c in zone_chargers:
                if self._is_vehicle_on_charger(v, c):
                    target_charger = c
                    break
                if any(p.vehicle is None for p in c.ports):
                    target_charger = c

            available_for_grid.append((idx, v, target_charger, c_v_norm))

        # =====================================================================
        # PHASE 2: ACTION PROJECTION (HARD 500 kW CEILING)
        # =====================================================================
        if peak_active:
            for _, v, tc, c_v_norm in available_for_grid:
                if v.battery.soc > 0.40 and c_v_norm < -0.05:
                    agent_alignment_bonus += 0.01
                if v.charger and v.status != VehicleStatus.DISCHARGING:
                    v.charger.disconnect(v.vid)
                if v.status != VehicleStatus.DISCHARGING:
                    v.status = VehicleStatus.IDLE

            remaining_v2b_kw = current_bldg_load - BUILDING_PEAK_LIMIT
            v2b_candidates = [
                item for item in available_for_grid if item[1].battery.soc > 0.35 and self.enable_v2g
            ]
            v2b_candidates.sort(key=lambda x: (x[3], -x[1].battery.soc))

            for _, v, tc, _ in v2b_candidates:
                if remaining_v2b_kw <= 0.05:
                    break
                alloc_kw = min(self.c_max, remaining_v2b_kw)
                eff_kw, rev = self._discharge_vehicle_v2b(v, tc, alloc_kw, eta, price_per_kwh)
                manual_v2g_power -= eff_kw
                remaining_v2b_kw = max(0.0, remaining_v2b_kw - eff_kw)
                step_v2b_revenue += rev

        else:
            remaining_charge_headroom_kw = max(0.0, (BUILDING_PEAK_LIMIT - 10.0) - current_bldg_load)
            available_for_grid.sort(key=lambda x: x[1].battery.soc)

            for _, v, tc, c_v_norm in available_for_grid:
                if v.battery.soc < 0.75 and c_v_norm > 0.05:
                    agent_alignment_bonus += 0.01

                if v.battery.soc >= 0.85:
                    if v.charger:
                        v.charger.disconnect(v.vid)
                    v.status = VehicleStatus.IDLE
                    continue

                if v.battery.soc < 0.55:
                    desired_kw = self.c_max
                else:
                    if c_v_norm < -0.35:
                        desired_kw = 0.0
                    else:
                        agent_frac = np.clip((c_v_norm + 1.0) / 2.0, 0.35, 1.0)
                        desired_kw = agent_frac * self.c_max

                if desired_kw > 1.0 and remaining_charge_headroom_kw >= self.c_max:
                    final_kw = min(self.c_max, desired_kw)
                    is_plugged = self._is_vehicle_on_charger(v, tc)
                    if is_plugged:
                        v.preferred_rate = final_kw
                        v.status = VehicleStatus.CHARGING
                    elif v.status == VehicleStatus.TOCHARGE and v.charger == tc:
                        v.preferred_rate = final_kw
                    else:
                        v.charge(tc, final_kw)
                    remaining_charge_headroom_kw -= self.c_max
                else:
                    if v.charger:
                        v.charger.disconnect(v.vid)
                    v.status = VehicleStatus.IDLE

        # =====================================================================
        # PHASE 3: PHYSICS TICK & POST-TICK PEAK GUARANTEE
        # =====================================================================
        for vehicle in self.fleet:
            vehicle.tick(self.dt, {'T_a': self.T_a})

        if peak_active:
            for charger in self.charging_network:
                for port in charger.ports:
                    if port.vehicle is not None:
                        vid = port.vehicle if isinstance(port.vehicle, int) else port.vehicle.vid
                        v_obj = self.fleet[vid]
                        # CRITICAL FIX: Do not unplug cars actively saving the grid
                        if v_obj.status != VehicleStatus.DISCHARGING:
                            charger.disconnect(vid)

        for charger in self.charging_network:
            charger.tick(self.fleet, self.dt, self.T_a)
            for port in charger.ports:
                if port.vehicle and port.P_t > 0:
                    self.current_grid_power += port.P_t

        prelim_net_load = current_bldg_load + self.current_grid_power + manual_v2g_power
        if prelim_net_load > BUILDING_PEAK_LIMIT and self.enable_v2g:
            residual_kw = prelim_net_load - BUILDING_PEAK_LIMIT
            reserve_v2b = [
                (v, tc)
                for _, v, tc, _ in available_for_grid
                if v.status == VehicleStatus.IDLE and v.battery.soc > 0.35
            ]
            reserve_v2b.sort(key=lambda x: -x[0].battery.soc)
            for v, tc in reserve_v2b:
                if residual_kw <= 0.05:
                    break
                alloc_kw = min(self.c_max, residual_kw)
                eff_kw, rev = self._discharge_vehicle_v2b(v, tc, alloc_kw, eta, price_per_kwh)
                manual_v2g_power -= eff_kw
                residual_kw = max(0.0, residual_kw - eff_kw)
                step_v2b_revenue += rev

        self.current_grid_power += manual_v2g_power

        # =====================================================================
        # PHASE 4: DEMAND QUEUE & REWARD COMPUTATION
        # =====================================================================
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
        current_capacity = sum(v.battery.actual_capacity for v in self.fleet)
        step_soh_loss_kwh = max(0.0, self.prev_capacity - current_capacity)
        self.prev_capacity = current_capacity

        net_building_load = current_bldg_load + self.current_grid_power
        violation = max(0.0, net_building_load - BUILDING_PEAK_LIMIT)
        baseline_violation = max(0.0, current_bldg_load - BUILDING_PEAK_LIMIT)
        peak_shaved_kw = max(0.0, baseline_violation - violation)

        reward = 0.0
        reward += 2.0 * len(to_completed)
        reward += 1.5 * (peak_shaved_kw / 100.0)
        if peak_active and violation < 1.0:
            reward += 1.0
        reward += agent_alignment_bonus
        reward -= 0.5 * step_soh_loss_kwh
        reward -= 2.0 * (violation / 100.0)

        class FastInfo(dict):
            def __deepcopy__(self, memo):
                return self

        info = FastInfo({
            'arrived': [j.to_dict() for j in self.arrived],
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
            'peak_violation': violation,
        })

        return self._get_obs(), float(reward), self.t >= self.t_max, False, info