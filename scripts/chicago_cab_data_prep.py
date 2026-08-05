"""Clean the Chicago city dataset."""
import argparse
import logging
import os
import glob

import coloredlogs
import numpy
import pandas

LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clean the Chicago city dataset.')
    parser.add_argument(
        '--raw-dir',
        '-r',
        required=True,
        help='Path to the folder containing raw parquet files.'
    )
    parser.add_argument(
        '--verbosity',
        '-v',
        choices=['debug', 'info', 'warn', 'error'],
        default='debug',
        help='Logging verbosity.'
    )
    args = parser.parse_args()
    coloredlogs.install(level=args.verbosity.upper())

    # Dynamically find all parquet files in the specified folder
    parquet_files = sorted(glob.glob(os.path.join(args.raw_dir, '*.parquet')))
    if not parquet_files:
        LOGGER.error(f"No .parquet files found in directory: {args.raw_dir}")
        exit(1)

    data = []
    LOGGER.debug('Loading raw data...')
    for file in parquet_files:
        LOGGER.debug(f'Loading file: {file}...')
        data.append(pandas.read_parquet(file))
    data = pandas.concat(data)

    LOGGER.debug('Dropping unneeded columns...')
    data.drop(
        columns=[
            'trip_id', 'taxi_id', 'trip_seconds', 'pickup_census_tract',
            'dropoff_census_tract', 'tips', 'tolls', 'extras',
            'payment_type', 'company', 'pickup_centroid_longitude',
            'pickup_centroid_latitude', 'pickup_centroid_location',
            'dropoff_centroid_longitude', 'dropoff_centroid_latitude',
            'dropoff_centroid_location',
            'Trip ID', 'Taxi ID', 'Trip Seconds', 'Pickup Census Tract',
            'Dropoff Census Tract', 'Tips', 'Tolls', 'Extras',
            'Payment Type', 'Company', 'Pickup Centroid LongLongitude',
            'Pickup Centroid Latitude', 'Pickup Centroid Location',
            'Dropoff Centroid Longitude', 'Dropoff Centroid Latitude',
            'Dropoff Centroid  Location'
        ],
        errors='ignore',
        inplace=True
    )

    LOGGER.debug('Dropping NAN and INF values...')
    data.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
    data.dropna(inplace=True)

    LOGGER.debug('Standardizing column names...')
    data.rename(
        columns={
            'trip_start_timestamp': 'pickup_time',
            'trip_end_timestamp': 'dropoff_time',
            'trip_miles': 'distance',
            'pickup_community_area': 'pickup_location',
            'dropoff_community_area': 'dropoff_location',
            'trip_total': 'fare',
            'Trip Start Timestamp': 'pickup_time',
            'Trip End Timestamp': 'dropoff_time',
            'Trip Miles': 'distance',
            'Pickup Community Area': 'pickup_location',
            'Dropoff Community Area': 'dropoff_location',
            'Trip Total': 'fare',
        },
        inplace=True
    )

    LOGGER.debug('Casting data to correct types...')
    data['pickup_time'] = pandas.to_datetime(data['pickup_time'], format='mixed')
    data['dropoff_time'] = pandas.to_datetime(data['dropoff_time'], format='mixed')
    data['distance'] = 1.6 * data['distance'].astype(str).str.replace(',', '').astype(float)
    data['pickup_location'] = data['pickup_location'].astype(int)
    data['dropoff_location'] = data['dropoff_location'].astype(int)
    data['fare'] = data['fare'].astype(str).str.replace(',', '').str.replace('$', '').astype(float)

    LOGGER.debug('Dropping nonsensical data...')
    data.drop(data[data['pickup_time'] >= data['dropoff_time']].index, inplace=True)
    data.drop(data[data['distance'] <= 0].index, inplace=True)
    data.drop(data[data['fare'] <= 0].index, inplace=True)

    LOGGER.debug('Sorting demand by pickup time...')
    data.sort_values(by='pickup_time', ascending=True, inplace=True)

    LOGGER.debug('Writing to file...')
    data.to_csv('chicago_demand.csv', index=False)

    LOGGER.info('Successfully cleaned Chicago cab data.')