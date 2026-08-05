"""Clean the Chicago city dataset."""
import argparse
import logging


import coloredlogs
import numpy
import pandas


DATE_FORMAT = '%m/%d/%Y %I:%M:%S %p'
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clean the Chicago city dataset.')
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

    # LOGGER.debug('Dropping unneeded columns...')
    # data.drop(
    #     columns=[
    #         'Trip ID',
    #         'Taxi ID',
    #         'Trip Seconds',
    #         'Pickup Census Tract',
    #         'Dropoff Census Tract',
    #         'Fare',
    #         'Tips',
    #         'Tolls',
    #         'Extras',
    #         'Payment Type',
    #         'Company',
    #         'Pickup Centroid Longitude',
    #         'Pickup Centroid Latitude',
    #         'Pickup Centroid Location',
    #         'Dropoff Centroid Longitude',
    #         'Dropoff Centroid Latitude',
    #         'Dropoff Centroid  Location',
    #     ],
    #     inplace=True
    # )

    LOGGER.debug('Dropping unneeded columns...')
    data.drop(
        columns=[
            # New snake_case headers
            'trip_id', 'taxi_id', 'trip_seconds', 'pickup_census_tract',
            'dropoff_census_tract', 'fare', 'tips', 'tolls', 'extras',
            'payment_type', 'company', 'pickup_centroid_longitude',
            'pickup_centroid_latitude', 'pickup_centroid_location',
            'dropoff_centroid_longitude', 'dropoff_centroid_latitude',
            'dropoff_centroid_location',
            # Old Title Case headers (fallback)
            'Trip ID', 'Taxi ID', 'Trip Seconds', 'Pickup Census Tract',
            'Dropoff Census Tract', 'Fare', 'Tips', 'Tolls', 'Extras',
            'Payment Type', 'Company', 'Pickup Centroid Longitude',
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

    # LOGGER.debug('Standardizing column names...')
    # data.rename(
    #     columns={
    #         'Trip Start Timestamp': 'pickup_time',
    #         'Trip End Timestamp': 'dropoff_time',
    #         'Trip Miles': 'distance',
    #         'Pickup Community Area': 'pickup_location',
    #         'Dropoff Community Area': 'dropoff_location',
    #         'Trip Total': 'fare',
    #     },
    #     inplace=True
    # )

    LOGGER.debug('Standardizing column names...')
    data.rename(
        columns={
            # New snake_case headers
            'trip_start_timestamp': 'pickup_time',
            'trip_end_timestamp': 'dropoff_time',
            'trip_miles': 'distance',
            'pickup_community_area': 'pickup_location',
            'dropoff_community_area': 'dropoff_location',
            'trip_total': 'fare',
            # Old Title Case headers (fallback)
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
    # data['pickup_time'] = pandas.to_datetime(data['pickup_time'], format=DATE_FORMAT)
    # data['dropoff_time'] = pandas.to_datetime(data['dropoff_time'], format=DATE_FORMAT)

    # Change format=DATE_FORMAT to format='mixed'
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
