"""Models for charging infrastructure."""

from typing import Dict, List, Union


from simulator.region import *
from simulator.vehicle import *


class ChargePort:
    """Single port in a charging station.

    Args:
        P_max - maximum instantaneous supply power (kW)
        efficiency - charging efficiency (%)
    """

    def __init__(self, P_max: float, efficiency: float) -> None:
        self.P_max = P_max
        self.efficiency = efficiency
        self.vehicle = None
        self.P_t = 0

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Express this object as a dictionary.

        Returns:
            {
                P_max: maximum charging power allowed,
                P_t: current charging power,
                efficiency: charging efficiency (%),
                vehicle: vehicle id (None if no vehicle present),
            }
        """
        return {
            "P_max": self.P_max,
            "P_t": self.P_t,
            "efficiency": self.efficiency,
            "vehicle": self.vehicle,
        }


class ChargeStation:
    """Charging station.

    Args:
        location: map loacation
        ports: list of charging ports that comprise the station
        P_max: maximum power output of the station (kW) (None indicates this
            is the sum of P_max for all ports)
        queue_size: maximum number of vehicles that can wait to charge (None
            means unbounded)
    """

    def __init__(
        self, location: Location, ports: List[ChargePort], P_max: float = None
    ) -> None:
        self.location = location
        self.ports = ports
        self.P_max = P_max
        self.vehicle_queue = {}

    def to_dict(self) -> Dict[str, Union[Dict, List, int, float]]:
        """Return representing the current charging station state.

        Returns:
            {
                location: charging station location,
                ports: list of ports and their current state,
                P_max: maximum power output,
                vheicle_queue: list of waiting vehiclde ids
            }
        """
        return {
            "location": self.location.to_dict(),
            "ports": [p.to_dict() for p in self.ports],
            "P_max": self.P_max,
            "vehicle_queue": [vid for vid in self.vehicle_queue],
        }

    def request_charge(self, preferred_rate: float, vehicle: int) -> None:
        """
        A <vehicle> requests a maximum charge rate <preferred rate> in kW.
        The requested rate may not be provided, but will never be exceeded.
        """
        for port in self.ports:
            if port.vehicle == vehicle:
                port.P_t = min(preferred_rate, port.P_max)
                return
        self.vehicle_queue[vehicle] = preferred_rate

    def disconnect(self, vehicle: int) -> None:
        """
        Disconnect <vehicle> from this charging station.
        """
        for port in self.ports:
            if port.vehicle == vehicle:
                port.vehicle = None
                port.P_t = 0.0
                return
        if vehicle in self.vehicle_queue:
            del self.vehicle_queue[vehicle]

    def tick(self, fleet: List, dt: float, T_a: float) -> None:
        """
        Update the state of all vehicles currently charging.

        Args:
            fleet: global list of vehicles.
            dt: the tick length across which to recalculate state.
            T_a: the ambient temperature of the charging station on the tick
                interval.
        """
        to_charge = list(self.vehicle_queue.keys())
        power_requested = 0.0
        for port in self.ports:
            if port.vehicle is None and len(to_charge) > 0:
                vehicle = to_charge.pop()
                port.vehicle = vehicle
                port.P_t = min(self.vehicle_queue[vehicle], port.P_max)
                del self.vehicle_queue[vehicle]
            if port.vehicle is not None:
                if power_requested + port.P_t <= self.P_max:
                    power_requested += port.P_t
                else:
                    port.P_t = max(0.0, self.P_max - power_requested)
                    power_requested += port.P_t
        for port in self.ports:
            if port.vehicle is not None:
                fleet[port.vehicle].battery.charge(port.P_t, dt, T_a)
