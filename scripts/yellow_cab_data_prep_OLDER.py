"""Clean the New York city dataset."""
import argparse
import logging


import coloredlogs
import numpy
import pandas


DATE_FORMAT = '%m/%d/%Y %I:%M:%S %p'
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clean the NYC Yellow Cab dataset.')
    parser.add_argument(
        '--raw-data',
        '-r',
        nargs='+',
        help='List of csv files containing raw demand data.'
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

    data = []
    LOGGER.debug('Loading raw data...')
    # for file in args.raw_data:
    #     LOGGER.debug(f'Loading file: {DATA}...')
    #     data.append(pandas.read_csv(DATA))

    for file in args.raw_data:
        LOGGER.debug(f'Loading file: {file}...')  # Fixed from DATA
        data.append(pandas.read_csv(file))        # Fixed from DATA
        
    data = pandas.concat(data)

    LOGGER.debug('Dropping unneeded columns...')
    data.drop(
        columns=[
            'VendorID',
            'RatecodeID',
            'store_and_fwd_flag',
            'payment_type',
            'fare_amount',
            'extra',
            'mta_tax',
            'tip_amount',
            'tolls_amount',
            'improvement_surcharge',
            'congestion_surcharge',
        ],
        inplace=True
    )

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
    data['pickup_time'] = pandas.to_datetime(data['pickup_time'], format=DATE_FORMAT)
    data['dropoff_time'] = pandas.to_datetime(data['dropoff_time'], format=DATE_FORMAT)
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
