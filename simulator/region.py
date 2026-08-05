"""Region map."""

from typing import Dict, ForwardRef, Tuple


import pickle


class Location:
    """Abstract class representing a location in a region."""

    def __init__(self, region: ForwardRef("Region")) -> None:
        self.region = region

    def to_dict(self) -> Dict:
        """Represent the location as a dictionary."""
        raise NotImplemented

    def to(self, location: "Location") -> Tuple[float, float]:
        """Distance to another <location> in the same map.

        Returns:
            (distance, time) in km and seconds respectively.
        """
        return self.region.distance(self, location)


class Region:
    """Abstract class for a region.  A region acts as a map, recording the
    time, distance, and travel conditions between pickups, dropoffs, and
    charging stations.  A Region must implement the distance method, which
    returns the distance between two locations.
    """

    def __init__(self) -> None:
        pass

    def distance(
        self, start: Location, end: Location, conditions: Dict = None
    ) -> Tuple[float, float]:
        """Calculate the distance between <start> and <end> given <conditions>.

        Args:
            start: starting location
            end: ending location
            conditions: environmental conditions

        Returns:
            (distance, time) in km and seconds respectively.
        """
        raise NotImplemented


class CyclicZoneGraphLocation(Location):
    """Location in a cyclic zone graph.

    Args:
        zone: node number within graph.
    """

    def __init__(self, zone: int, region: Region) -> None:
        super().__init__(region)
        self.zone = zone

    def to_dict(self) -> Dict:
        """Represent the location as a dictionary."""
        return self.zone

    def to(self, location: Location) -> Tuple[float, float]:
        """Distance to another <location> in the same map.

        Returns:
            (distance, time) in km and seconds respectively.
        """
        return self.region.distance(self, location)


class CyclicZoneGraph(Region):
    """Region comprised of zones connected by bidirectional edges.

    Args:
        mapfile: path to file containing map data.
    """

    def __init__(self, mapfile: str) -> None:
        super().__init__()
        with open(mapfile, "rb") as pklfile:
            self.map = pickle.loads(pklfile.read())

    def distance(
        self, start: Location, end: Location, conditions: Dict = None
    ) -> float:
        """Calculate the distance between <start> and <end> given <conditions>.

        Args:
            start: starting location
            end: ending location
            conditions: environmental conditions

        Returns:
            (distance, time) in km and seconds respectively.
        """
        return (
            self.map[start.zone][end.zone]["distance"],
            self.map[start.zone][end.zone]["time"],
        )
