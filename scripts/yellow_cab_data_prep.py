"""Clean the New York city dataset (Memory-Safe Version)."""
import argparse
import logging
import os
import glob

import coloredlogs
import numpy as np
import pandas as pd

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

    parquet_files = sorted(glob.glob(os.path.join(args.raw_dir, '*.parquet')))
    if not parquet_files:
        LOGGER.error(f"No .parquet files found in directory: {args.raw_dir}")
        exit(1)

    needed_columns = [
        'tpep_pickup_datetime', 
        'tpep_dropoff_datetime', 
        'passenger_count',
        'trip_distance', 
        'PULocationID', 
        'DOLocationID', 
        'total_amount'
    ]

    output_file = 'nyc_demand.csv'
    # Remove existing output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    LOGGER.debug('Processing raw files incrementally...')
    first_chunk = True

    for file in parquet_files:
        LOGGER.debug(f'Processing file: {file}...')
        try:
            chunk = pd.read_parquet(file, columns=needed_columns)
        except ValueError as e:
            LOGGER.warn(f'Skipping {file} due to missing columns: {e}')
            continue

        LOGGER.debug('Standardizing column names...')
        chunk.rename(
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

        LOGGER.debug('Dropping NAN and INF values...')
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(subset=['pickup_location', 'dropoff_location', 'distance', 'fare'], inplace=True)

        LOGGER.debug('Filtering out invalid TLC zones...')
        chunk = chunk[chunk['pickup_location'].between(1, 263)]
        chunk = chunk[chunk['dropoff_location'].between(1, 263)]

        LOGGER.debug('Casting data to correct types...')
        chunk['pickup_time'] = pd.to_datetime(chunk['pickup_time'], format='mixed')
        chunk['dropoff_time'] = pd.to_datetime(chunk['dropoff_time'], format='mixed')
        chunk['passenger_count'] = chunk['passenger_count'].fillna(1).astype(int)
        chunk['distance'] = 1.6 * chunk['distance'].astype(float)
        chunk['pickup_location'] = chunk['pickup_location'].astype(int)
        chunk['dropoff_location'] = chunk['dropoff_location'].astype(int)
        chunk['fare'] = chunk['fare'].astype(float)

        LOGGER.debug('Dropping nonsensical data...')
        chunk.drop(chunk[chunk['pickup_time'] >= chunk['dropoff_time']].index, inplace=True)
        chunk.drop(chunk[chunk['passenger_count'] < 1].index, inplace=True)
        chunk.drop(chunk[chunk['distance'] <= 0].index, inplace=True)
        chunk.drop(chunk[chunk['fare'] <= 0].index, inplace=True)

        LOGGER.debug('Sorting chunk by pickup time...')
        chunk.sort_values(by='pickup_time', ascending=True, inplace=True)

        LOGGER.debug(f'Appending {file} to {output_file}...')
        # Write incrementally to CSV
        if first_chunk:
            chunk.to_csv(output_file, mode='w', index=False)
            first_chunk = False
        else:
            chunk.to_csv(output_file, mode='a', header=False, index=False)

    # Final global sort if absolute chronological order across all files is strictly required:
    # Note: If the file is too large to re-read fully for sorting, you can use an external sort or process monthly.
    LOGGER.info('Successfully cleaned and saved NYC yellow cab data incrementally.')