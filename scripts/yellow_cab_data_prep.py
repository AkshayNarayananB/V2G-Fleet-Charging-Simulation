"""Clean the New York city dataset."""
import argparse
import logging
import os
import glob

import coloredlogs
import numpy
import pandas

LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clean the NYC Yellow Cab dataset.')
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

    # Define only the columns the simulator actually needs
    needed_columns = [
        'tpep_pickup_datetime', 
        'tpep_dropoff_datetime', 
        'passenger_count',
        'trip_distance', 
        'PULocationID', 
        'DOLocationID', 
        'total_amount'
    ]

    data = []
    LOGGER.debug('Loading raw data (memory-optimized)...')
    for file in parquet_files:
        LOGGER.debug(f'Loading file: {file}...')
        # Only load the specified columns from the parquet file to save RAM
        try:
            chunk = pandas.read_parquet(file, columns=needed_columns)
            data.append(chunk)
        except ValueError as e:
            LOGGER.warn(f'Skipping {file} due to missing columns: {e}')
            
    LOGGER.debug('Concatenating data...')
    data = pandas.concat(data, ignore_index=True)

    LOGGER.debug('Dropping NAN and INF values...')
    data.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
    data.dropna(inplace=True)

    LOGGER.debug('Dropping NAN and INF values...')
    data.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
    data.dropna(inplace=True)

    LOGGER.debug('Standardizing column names...')
    data.rename(
        columns={
            'tpep_pickup_datetime': 'pickup_time',
            'tpep_dropoff_datetime': 'dropoff_time',
            'passenger_count': 'passenger_count',
            'trip_distance': 'distance',
            'PULocationID': 'pickup_location',
            'DOLocationID': 'dropoff_location',
            'total_amount': 'fare',
        },
        inplace=True
    )

    LOGGER.debug('Casting data to correct types...')
    data['pickup_time'] = pandas.to_datetime(data['pickup_time'], format='mixed')
    data['dropoff_time'] = pandas.to_datetime(data['dropoff_time'], format='mixed')
    data['passenger_count'] = data['passenger_count'].astype(int)
    data['distance'] = 1.6 * data['distance'].astype(float)
    data['pickup_location'] = data['pickup_location'].astype(int)
    data['dropoff_location'] = data['dropoff_location'].astype(int)
    data['fare'] = data['fare'].astype(float)

    LOGGER.debug('Dropping nonsensical data...')
    data.drop(data[data['pickup_time'] >= data['dropoff_time']].index, inplace=True)
    data.drop(data[data['passenger_count'] < 1].index, inplace=True)
    data.drop(data[data['distance'] <= 0].index, inplace=True)
    data.drop(data[data['fare'] <= 0].index, inplace=True)

    LOGGER.debug('Sorting demand by pickup time...')
    data.sort_values(by='pickup_time', ascending=True, inplace=True)

    LOGGER.debug('Writing to file...')
    data.to_csv('nyc_demand.csv', index=False)

    LOGGER.info('Successfully cleaned NYC yellow cab data.')