"""Battery models."""

from typing import Dict


import math


class BatteryOverChargeException(Exception):
    """More power supplied than battery can handle.

    Args:
        message: custom error message
        surplus: amount of surplus energy (kWh)
    """

    def __init__(self, message: str, surplus: float) -> None:
        super.__init__(message)
        self.surplus = surplus


class BatteryEmptyException(Exception):
    """More power requested than available.

    Args:
        message: custom error message
        deficit: extra energy needed to complete request (kWh)
    """

    def __init__(self, message: str, deficit: float) -> None:
        super.__init__(message)
        self.deficit = deficit


class Battery:
    """Abstract battery class.

    Inheriting classes must implement the charge and discharge methods, and,
    optionally the age method.

    Args:
        capacity: battery capcity in kWh
    """

    def __init__(self, capacity: float) -> None:
        self.initial_capacity = capacity
        self.actual_capacity = capacity
        self.soc = 1

    def to_dict(self) -> Dict[str, float]:
        """Return a dictionary representing the current state of the battery.

        Returns:
            {
                initial_capacity: capacity when the battery was new,
                actual_capacity: current capacity of the battery,
                soc: state of charge (%),
            }
        """
        return {
            "initial_capacity": self.initial_capacity,
            "actual_capacity": self.actual_capacity,
            "soc": self.soc,
        }

    def charge(self, dW: float, dt: float, T_a: float) -> None:
        """Simulate charging the battery with <dW> kWh across <dt> seconds at
        <T_a> degrees Celsius.

        Args:
            dW: charge energy in kWh
            dt: charging time (seconds)
            T_a: battery temperature (Celsius)

        Raises:
            BatteryOverChargeException
        """
        raise NotImplemented

    def discharge(self, dW: float, dt: float, T_a: float) -> None:
        """Simulate discharge of <dW> kWh across <dt> seconds at <T_a> degrees
        Celsius.

        Args:
            dW: discharge energy in kWh
            dt: discharge time (seconds)
            T_a: battery temperature (Celsius)
        Raises:
            BatteryEmptyException
        """
        raise NotImplemented

    def age(self, dt: float, T_a: float) -> None:
        """Simulate battery aging for <dt> seconds. at <T_a> degrees Celsius."""
        raise NotImplemented


class MultiStageBattery(Battery):
    """This battery is affected by two types of aging:
        Cyclic: Wan et al.
        Calendar: ?

    Assumptions:
        1. Charge / discharge efficiency is 100%
        2. No loss of SoC in storage
    """

    def __init__(self, capacity: float) -> None:
        super().__init__(capacity)

    def recalculate_capacity(self, dW, dt, T_a) -> None:
        """Under Wan et al. cyclic aging due to charging and discharging is
        equivalent.  This method recalculates SoH and SoC for any inflow /
        outflow of energy dW.

        Args:
            dW - energy in (+) or out (-) in kWh
            dt - time (seconds)
            T_a - ambient temperature in Celcius.

        Raises:
            BatteryOverChargeException, BatteryEmptyException
        """
        if self.actual_capacity / self.initial_capacity > 0.933:
            alpha = 0.2172
            beta = 24.2535
            psi = -12.0051
            zeta = 0.3952
        elif self.actual_capacity / self.initial_capacity > 0.866:
            alpha = 0.2652
            beta = 9.9653
            psi = -29.0049
            zeta = 0.4470
        else:
            alpha = 0.2611
            beta = -15.1963
            psi = -22.5247
            zeta = 0.5066

        DoD_ref = 1.0
        DoD_t = (self.soc * self.actual_capacity + dW) / self.actual_capacity
        # if DoD_t < 0 or DoD_t > 1:
        # raise Exception(f'Magnitude of delta W too large: SoC - {self.soc}; Cap. - {self.actual_capacity}; W - {delta_W}')
        if DoD_t <= 0:
            DoD_t = 0.0
            dW = self.actual_capacity
        if DoD_t >= 1:
            DoD_t = 1.0
            dW = (1 - self.soc) * self.actual_capacity
        C = self.initial_capacity
        I_ref = 0.5 * C
        I_t = dW / (dt / 3600)
        T_ref = 25

        if I_t <= 1e-5 and I_t >= -1e-5:
            # In the case where the current drawn is so small, don't don anything
            return

        theta_t = abs(
            (DoD_t / DoD_ref) ** (1 / alpha)
            * (I_t / I_ref) ** (1 / beta)
            * math.exp(-psi * (1 / T_a - 1 / T_ref))
        )
        N_cref = 513  # Wan et al. 2024 (Good for single and multistage)
        Q_loss = theta_t / N_cref
        assert Q_loss >= 0

        self.soc = DoD_t
        self.actual_capacity -= Q_loss
        if self.actual_capacity < 0:
            self.actual_capacity = 0

    def charge(self, dW: float, dt: float, T_a: float) -> None:
        """Simulate charging the battery with <dW> kWh across <dt> seconds at
        <T_a> degrees Celsius.

        Args:
            dW: charge energy in kWh
            dt: charging time (seconds)
            T_a: battery temperature (Celsius)

        Raises:
            BatteryOverChargeException
        """
        self.recalculate_capacity(dW, dt, T_a)

    def discharge(self, dW: float, dt: float, T_a: float) -> None:
        """Simulate discharge of <dW> kWh across <dt> seconds at <T_a> degrees
        Celsius.

        Args:
            dW: discharge energy in kWh
            dt: discharge time (seconds)
            T_a: battery temperature (Celsius)
        Raises:
            BatteryEmptyException
        """
        self.recalculate_capacity(-dW, dt, T_a)

    def age(self, dt: float, T_a: float) -> None:
        """Simulate battery aging for <dt> seconds. at <T_a> degrees Celsius."""
        return
