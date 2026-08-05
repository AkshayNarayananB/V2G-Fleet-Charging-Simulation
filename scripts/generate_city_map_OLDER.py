"""Generate a map of travel times between districts in a city given a dataset."""


import argparse
import csv
import datetime
import json
import logging
import pickle


import coloredlogs
import numpy


from scipy import stats


DATEFMT = '%Y-%m-%d %H:%M:%S'
LOGGER = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')


def dijkstra(loc, dest, MAP):
    dist = []
    prev = []
    Q = set()
    for v in MAP:
        dist.append(float('inf'))
        prev.append(None)
        Q.add(v)
    dist[loc - 1] = 0

    while len(Q) > 0:
        u = list(Q)[0]
        last_min = dist[u-1]
        for q in Q:
            if dist[q-1] < last_min:
                last_min = dist[q-1]
                u = q
        Q.remove(u)

        for v in Q:
            if MAP[u][v]['distance'] is not None:
                dv = MAP[u][v]['distance']
                alt = dist[u-1] + dv
                if alt < dist[v-1]:
                    dist[v-1] = alt
                    prev[v-1] = u

    S = []
    u = dest
    t = 0
    d = 0
    if prev[u-1] is not None or u == loc:
        while prev[u - 1] is not None:
            t += MAP[prev[u-1]][u]['time']
            d += MAP[prev[u-1]][u]['distance']
            S.append(u)
            u = prev[u-1]

    if dist[u-1] == float('inf'):
        return float('inf'), float('inf')

    return t, d


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare a city map of travel times.')
    parser.add_argument(
        '--dataset',
        '-d',
        help='Dataset used to build city.'
    )
    parser.add_argument(
        '--n-zones',
        '-n',
        help='Number of zones in city',
        type=int
    )
    parser.add_argument(
        '--map-name',
        '-m',
        help='Output map name'
    )
    args = parser.parse_args()
    
    city = {}
    
    for zone_from in range(args.n_zones):
        city[zone_from] = {}
        for zone_to in range(args.n_zones):
            city[zone_from][zone_to] = {
                'time': [],
                'distance': []
            }

    with open(args.dataset, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pu_loc = int(row['pickup_location'])
            do_loc = int(row['dropoff_location'])
            distance = float(row['distance'])
            pu_time = datetime.datetime.strptime(row['pickup_time'], DATEFMT)
            do_time = datetime.datetime.strptime(row['dropoff_time'], DATEFMT)
            city[pu_loc][do_loc]['distance'].append(distance)
            city[pu_loc][do_loc]['time'].append((do_time - pu_time).total_seconds())

    for zone_from in city:
        LOGGER.debug(f'Processing Zone: {zone_from}')
        for zone_to in city:
            if len(city[zone_from][zone_to]['time']) > 1:
                city[zone_from][zone_to]['time'] = numpy.mean(city[zone_from][zone_to]['time']).item()
                city[zone_from][zone_to]['distance'] = numpy.mean(city[zone_from][zone_to]['distance']).item()
            else:
                city[zone_from][zone_to]['time'] = None
                city[zone_from][zone_to]['distance'] = None

    for zone_from in city:
        LOGGER.debug(f'Calculating Unknown Routes: {zone_from}')
        for zone_to in city:
            if city[zone_from][zone_to]['time'] is None:
                t, d = dijkstra(zone_from, zone_to, city)
                city[zone_from][zone_to]['time'] = float(t)
                city[zone_from][zone_to]['distance'] = float(d)

    LOGGER.debug(f'Removing invalid zones')
    removal_list = []
    for zone_from in city:
        invalid = True
        for zone_to in city:
            if city[zone_from][zone_to]['distance'] != float('inf') and zone_from != zone_to:
                invalid = False
        if invalid:
            removal_list.append(zone_from)
    for zone in removal_list:
        del city[zone]
        for zone_from in city:
            del city[zone_from][zone]

    with open(args.map_name, 'wb') as pklfile:
        pklfile.write(pickle.dumps(city))
