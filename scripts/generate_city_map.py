"""Generate a map of travel times between districts in a city given a dataset (Optimized)."""

import argparse
import csv
import datetime
import json
import logging
import pickle
import os
import multiprocessing
from functools import partial

import coloredlogs
import numpy
import pandas

DATEFMT = '%Y-%m-%d %H:%M:%S'
LOGGER = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')


def dijkstra_worker(source_zone, MAP_KEYS, MAP):
    """Worker function to compute Dijkstra for a single source zone."""
    dist = {v: float('inf') for v in MAP_KEYS}
    prev = {v: None for v in MAP_KEYS}
    Q = set(MAP_KEYS)
    dist[source_zone] = 0

    while len(Q) > 0:
        u = min(Q, key=lambda x: dist[x])
        if dist[u] == float('inf'):
            break
        Q.remove(u)

        for v in Q:
            if v in MAP[u] and MAP[u][v]['distance'] is not None:
                dv = MAP[u][v]['distance']
                alt = dist[u] + dv
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u

    # Reconstruct paths/costs back to all destinations
    results = {}
    for dest in MAP_KEYS:
        if dest == source_zone:
            continue
        u = dest
        t = 0
        d = 0
        if prev[u] is not None or u == source_zone:
            while prev[u] is not None:
                t += MAP[prev[u]][u]['time']
                d += MAP[prev[u]][u]['distance']
                u = prev[u]
        if dist[dest] == float('inf'):
            results[dest] = (float('inf'), float('inf'))
        else:
            results[dest] = (t, d)
    return source_zone, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare a city map of travel times.')
    parser.add_argument('--dataset', '-d', help='Dataset used to build city.')
    parser.add_argument('--n-zones', '-n', help='Number of zones in city', type=int)
    parser.add_argument('--map-name', '-m', help='Output map name')
    args = parser.parse_args()

    LOGGER.info('Loading dataset with Pandas for vectorization...')
    # Load dataset using pandas chunks or direct read to bypass slow csv reader loops
    df = pandas.read_csv(args.dataset, usecols=['pickup_location', 'dropoff_location', 'distance', 'pickup_time', 'dropoff_time'])

    LOGGER.info('Parsing datetimes and calculating trip durations...')
    df['pickup_time'] = pandas.to_datetime(df['pickup_time'], format='mixed')
    df['dropoff_time'] = pandas.to_datetime(df['dropoff_time'], format='mixed')
    df['time'] = (df['dropoff_time'] - df['pickup_time']).dt.total_seconds()

    # LOGGER.info('Aggregating routes via Pandas GroupBy...')
    # grouped = df.groupby(['pickup_location', 'dropoff_location']).agg(
    #     time=('time', 'mean'),
    #     distance=('distance', 'mean')
    # ).reset()

    LOGGER.info('Aggregating routes via Pandas GroupBy...')
    grouped = df.groupby(['pickup_location', 'dropoff_location']).agg(
        time=('time', 'mean'),
        distance=('distance', 'mean')
    ).reset_index()  # Fixed from .reset() to .reset_index()

    city = {}
    for zone in range(1, args.n_zones + 1):
        city[zone] = {}
        for zone_to in range(1, args.n_zones + 1):
            city[zone][zone_to] = {'time': None, 'distance': None}

    for _, row in grouped.iterrows():
        pu = int(row['pickup_location'])
        do = int(row['dropoff_location'])
        if pu in city and do in city:
            city[pu][do]['time'] = float(row['time'])
            city[pu][do]['distance'] = float(row['distance'])

    zones = list(city.keys())

    LOGGER.info('Calculating Unknown Routes using Multiprocessing Dijkstra...')
    # Identify missing routes that need pathfinding
    missing_pairs = 0
    for u in zones:
        for v in zones:
            if u != v and city[u][v]['time'] is None:
                missing_pairs += 1

    LOGGER.info(f'Found {missing_pairs} unknown routes. Computing in parallel...')

    # Use multiprocessing pool to parallelize Dijkstra across CPU cores
    worker_func = partial(dijkstra_worker, MAP_KEYS=zones, MAP=city)
    cpu_count = max(1, multiprocessing.cpu_count() - 1)
    
    with multiprocessing.Pool(processes=cpu_count) as pool:
        outputs = pool.map(worker_func, zones)

    for source_zone, res in outputs:
        for dest_zone, (t, d) in res.items():
            if city[source_zone][dest_zone]['time'] is None:
                city[source_zone][dest_zone]['time'] = t
                city[source_zone][dest_zone]['distance'] = d

    LOGGER.debug('Removing invalid zones')
    removal_list = []
    for zone_from in list(city.keys()):
        invalid = True
        for zone_to in list(city.keys()):
            if city[zone_from][zone_to]['distance'] != float('inf') and zone_from != zone_to:
                invalid = False
        if invalid:
            removal_list.append(zone_from)
    for zone in removal_list:
        del city[zone]
        for zone_from in city:
            if zone in city[zone_from]:
                del city[zone_from][zone]

    LOGGER.info(f'Writing map to {args.map_name}...')
    with open(args.map_name, 'wb') as pklfile:
        pklfile.write(pickle.dumps(city))
    
    LOGGER.info('Successfully generated optimized city map.')