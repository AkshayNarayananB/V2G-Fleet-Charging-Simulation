"""Jobs."""

from typing import Dict, Union
from enum import Enum


import datetime


from simulator.region import *
from simulator.vehicle import *


DATEFMT = "%Y-%m-%d %H:%M:%S"


class JobStatus(Enum):
    """
    Possible job states.
    """

    ARRIVED = 1  # Job created
    ASSIGNED = 2  # Job assigned to vehicle
    INPROGRESS = 3  # Job is being serviced
    REJECTED = 4  # Job not to be serviced
    COMPLETE = 5  # Job successfully completed
    FAILED = 6  # Job failed


class Job:
    """
    Job - a request for taxi service.

    Args:
        data: job data.  Must contain keys: pickup_location (int),
            dropoff_location (int), pickup_time (timestamp), dropoff_time
            (timestamp), and distance (float, km).
        job_id: a unique integer id for this job.
        region: global region map used to convert locations into location
            objects.
    """

    def __init__(self, data: Dict, job_id: int, region: Region) -> None:
        self.id = job_id
        self.pickup_location = CyclicZoneGraphLocation(
            int(data["pickup_location"]), region
        )
        self.dropoff_location = CyclicZoneGraphLocation(
            int(data["dropoff_location"]), region
        )
        self.duration = datetime.datetime.strptime(
            data["dropoff_time"], DATEFMT
        ) - datetime.datetime.strptime(data["pickup_time"], DATEFMT)
        self.distance = float(data["distance"])
        self.fare = float(data["fare"])
        self.vehicle = None
        self.status = JobStatus.ARRIVED
        self.elapsed_time = 0

    def to_dict(self) -> Dict[str, Union[Dict, float, int, str]]:
        """
        Return a dictionary representing the current state of the job.

        Returns:
            { pickup_location (Dict), dropoff_location (Dict), duration
            (float, sec.), distance (float, km), fare (float, $),
            vehicle (int), status (str), id (int) }
        """
        return {
            "pickup_location": self.pickup_location.to_dict(),
            "dropoff_location": self.dropoff_location.to_dict(),
            "duration": self.duration.total_seconds(),
            "distance": self.distance,
            "fare": self.fare,
            "vehicle": self.vehicle,
            "status": self.status.name,
            "id": self.id,
        }

    def assign_vehicle(self, vehicle: int) -> None:
        """
        Assign <vehicle> to this job.
        """
        self.status = JobStatus.ASSIGNED
        self.vehicle = vehicle

    def inprogress(self) -> None:
        """
        Set the job status to "INPROGRESS".
        """
        self.status = JobStatus.INPROGRESS

    def complete(self) -> None:
        """
        Set the job status to "COMPLETE".
        """
        self.status = JobStatus.COMPLETE

    def fail(self) -> None:
        """
        Set the job status to "FAILED".
        """
        self.status = JobStatus.FAILED

    def tick(self, dt: float) -> None:
        """
        Update this job's state.

        Args:
            dt: tick length in seconds.
        """
        self.elapsed_time += dt
        if self.status == JobStatus.ARRIVED:
            if self.elapsed_time > dt:
                self.status = JobStatus.REJECTED
