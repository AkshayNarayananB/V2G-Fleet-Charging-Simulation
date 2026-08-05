"""Model of an electric vehicle."""

from typing import Dict, ForwardRef, Union
from enum import Enum


from simulator.battery import *
from simulator.region import *


class VehicleStatus(Enum):
    """
    Vehicle states.
    """

    IDLE = 1
    TOPICKUP = 2
    TOCHARGE = 3
    CHARGING = 4
    TOLOC = 5
    ONJOB = 6
    OFFDUTY = 7
    RECOVERY = 8


class Vehicle:
    """Electric Vehicle.

    Args:
        model: select from previously defined vehicles: ['byd e6'] or provide
            a diction in the format {'capacity': kWh, 'efficiency': kWh/100km}
        battery: select from previously defined models: ['multistage'] or
            provide an object inheriting from Battery.
        location: starting location of the vehicle.
    """

    def __init__(
        self,
        model: Union[str, Dict[str, float]],
        battery: Union[str, Battery],
        location: Location,
        vid: int,
    ) -> None:
        self.model = model
        self.vid = vid
        self.charger = None
        if self.model.lower() == "byd e6":
            capacity = 71.7
            self.efficiency = 17.1
        else:
            capacity = model["capacity"]
            self.efficiency = model["efficiency"]

        if battery.lower() == "multistage":
            self.battery = MultiStageBattery(capacity)
        else:
            self.battery = battery

        self.depo = location
        self.location = location
        self.destination = location
        self.distance_remaining = 0.0
        self.time_remaining = 0.0
        self.time_elapsed = 0.0
        self.status = VehicleStatus.IDLE

    def to_dict(self) -> Dict[str, Union[Dict, float, str]]:
        """Return a dictionary representing the current state of the vehicle.

        Returns:
            {
                location: vehicle's current location in the region,
                destination: vehicle's current destination (same as location
                    if vehicle is not travelling),
                distance_remaining: distance to destination (km),
                time_remaining: time to destination (seconds),
                status: vehicle's current state,
                battery: the current state of the vehicle's battery,
                time_elapsed: time elapsed since the vehicle began travel
            }
        """
        return {
            "location": self.location.to_dict(),
            "destination": self.destination.to_dict(),
            "distance_remaining": self.distance_remaining,
            "time_remaining": self.time_remaining,
            "status": self.status.name,
            "battery": self.battery.to_dict(),
            "time_elapsed": self.time_elapsed,
        }

    def service_demand(self, job: ForwardRef("Job")) -> None:
        """
        Assign a vehicle to <job>.
        """
        if self.charger:
            self.charger.disconnect(self.vid)
        self.destination = job.pickup_location
        self.time_remaining = self.location.to(self.destination)[1]
        self.job = job
        self.job.assign_vehicle(self.vid)
        self.status = VehicleStatus.TOPICKUP

    def charge(
        self, charger: ForwardRef("ChargeStation"), preferred_rate: float
    ) -> None:
        """
        Assign a vehicle to a <charger> and attempt to charge at
        <preferred_rate> (kW). The preferred rate is not guaranteed, but
        cannot be exceeded during charging.
        """
        self.charger = charger
        self.destination = charger.location
        self.time_remaining = self.location.to(self.destination)[1]
        self.preferred_rate = preferred_rate

        if self.status == VehicleStatus.CHARGING and self.charger == charger:
            return

        if self.status != VehicleStatus.CHARGING or self.location != self.destination:
            self.status = VehicleStatus.TOCHARGE
            self.charger.disconnect(self.vid)

    def initialize_recovery_state(self) -> None:
        """
        Set the vehicle to return to the depot fully charge after a 24 hour
        timeout period.
        """
        self.destination = self.depo
        self.time_remaining = 24 * 60 * 60
        self.battery.charge(self.battery.actual_capacity, 3600, T_a=25)

    def tick(self, dt: float, conditions: Dict[str, int]) -> None:
        """
        Update the vehicle's state.

        Args:
            dt: tick length (seconds).
            conditions: environmental conditions present during the tick.
        """
        if self.status == VehicleStatus.IDLE:
            self.battery.age(dt, conditions["T_a"])
        elif self.status == VehicleStatus.TOPICKUP:
            self.time_remaining -= dt
            if self.time_remaining <= 0:
                distance = self.location.to(self.destination)[0]
                dW = distance * self.efficiency / 100
                self.battery.discharge(dW, dt, conditions["T_a"])
                if self.battery.soc <= 0:
                    self.status = VehicleStatus.RECOVERY
                    self.job.fail()
                    self.initialize_recovery_state()
                else:
                    self.destination = self.job.dropoff_location
                    self.time_remaining = self.location.to(self.destination)[1]
                    self.job.inprogress()
                    self.status = VehicleStatus.ONJOB
        elif self.status == VehicleStatus.TOCHARGE:
            self.time_remaining -= dt
            if self.time_remaining <= 0:
                distance = self.location.to(self.destination)[0]
                dW = distance * self.efficiency / 100
                self.battery.discharge(dW, dt, conditions["T_a"])
                if self.battery.soc <= 0:
                    self.status = VehicleStatus.RECOVERY
                    self.initialize_recovery_state()
                else:
                    self.status = VehicleStatus.CHARGING
        elif self.status == VehicleStatus.CHARGING:
            self.charger.request_charge(self.preferred_rate, self.vid)
        elif self.status == VehicleStatus.TOLOC:
            self.time_remaining -= dt
            if self.time_remaining <= 0:
                distance = self.location.to(self.destination)[0]
                dW = distance * self.efficiency / 100
                self.battery.discharge(dW, dt, conditions["T_a"])
                if self.battery.soc <= 0:
                    self.status = VehicleStatus.RECOVERY
                    self.initialize_recovery_state()
                else:
                    self.status = VehicleStatus.IDLE
        elif self.status == VehicleStatus.ONJOB:
            self.time_remaining -= dt
            if self.time_remaining <= 0:
                distance = self.location.to(self.destination)[0]
                dW = distance * self.efficiency / 100
                self.battery.discharge(dW, dt, conditions["T_a"])
                if self.battery.soc <= 0:
                    self.status = VehicleStatus.RECOVERY
                    self.job.fail()
                    self.initialize_recovery_state()
                else:
                    self.status = VehicleStatus.IDLE
                    self.job.complete()
        elif self.status == VehicleStatus.RECOVERY:
            self.time_remaining -= dt
            if self.time_remaining <= 0:
                self.status = VehicleStatus.IDLE
        else:
            raise Exception(f"Invalid vehicle state: {self.status}")
