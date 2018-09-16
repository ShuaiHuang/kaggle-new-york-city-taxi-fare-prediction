# -*- coding:utf-8 -*-
import os
import re
import logging
import argparse
import numpy as np
import pandas as pd
import datetime as dt
from pytz import timezone

DROP_FLAG_DEFAULT = 0
DROP_FLAG_SELECTED = -1
DROP_FLAG_INVALID_VALUE = 1
DROP_FLAG_INVALID_FARE = 2
DROP_FLAG_INVALID_LOCATION = 3
DROP_FLAG_INVALID_PASSENGER = 4
DROP_FLAG_INVALID_DATE = 5


def get_chunk_files(input_path, reg_exp):
    file_list = os.listdir(input_path)
    chunk_file_list = []
    for item in file_list:
        if re.match(reg_exp, item) is not None:
            chunk_file_list.append(item)
    chunk_file_list.sort()
    return chunk_file_list


def read_data_frame_from_file(input_file_path, input_data_formation=None):
    if input_data_formation == 'feather':
        df = pd.read_feather(input_file_path, nthreads=6)
    else:
        df = pd.read_csv(input_file_path)

    return df


def write_data_frame_to_file(data_frame, output_file_path, output_data_formation=None):
    if output_data_formation == 'feather':
        data_frame.to_feather(output_file_path)
    else:
        data_frame.to_csv(output_file_path, index=False)


def get_output_file_path(input_file_path, output_file_dir, output_data_formation=None):
    _, filename = os.path.split(input_file_path)

    if output_data_formation == 'feather':
        filename_base, _ = os.path.splitext(filename)
        filename = 'cleaned_%s.feather'%(filename_base,)
    else:
        filename = 'cleaned_%s'%(filename,)

    output_file_path = os.path.join(output_file_dir, filename)
    return output_file_path


def parse_date_time(data_frame):
    utc_timezone = timezone('utc')
    new_york_timezone = timezone('US/Eastern')
    data_frame['pickup_datetime_utc'] = data_frame['pickup_datetime'].map(lambda x: utc_timezone.localize(dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S %Z')))
    data_frame['pickup_datetime_local'] = data_frame['pickup_datetime_utc'].map(lambda x: x.astimezone(new_york_timezone))
    data_frame['pickup_year'] = data_frame['pickup_datetime_local'].map(lambda x: x.year)
    data_frame['pickup_month'] = data_frame['pickup_datetime_local'].map(lambda x: x.month)
    data_frame['pickup_day'] = data_frame['pickup_datetime_local'].map(lambda x: x.day)
    data_frame['pickup_hour'] = data_frame['pickup_datetime_local'].map(lambda x: x.hour)
    data_frame['pickup_minute'] = data_frame['pickup_datetime_local'].map(lambda x: x.minute)
    data_frame['pickup_second'] = data_frame['pickup_datetime_local'].map(lambda x: x.second)
    data_frame['pickup_weekday'] = data_frame['pickup_datetime_local'].map(lambda x: x.weekday())

    data_frame.loc[data_frame['pickup_year'] == 2008, 'drop_flag'] = DROP_FLAG_INVALID_DATE

    return data_frame


def extract_airport_location(data_frame):
    # New York JohnFitzgerald Kennedy International Airport
    data_frame['airport_jfk'] = 0
    data_frame.loc[(data_frame['pickup_longitude'].between(-73.7841, -73.7721)) & \
                   (data_frame['pickup_latitude'].between(40.6613, 40.6213)),'airport_jfk'] = 1
    data_frame.loc[(data_frame['dropoff_longitude'].between(-73.7841, -73.7721)) & \
                   (data_frame['dropoff_latitude'].between(40.6613, 40.6213)), 'airport_jfk'] = 1

    # LaGuardia Airport
    data_frame['airport_lga'] = 0
    data_frame.loc[(data_frame['pickup_longitude'].between(-73.8870, -73.8580)) & \
                   (data_frame['pickup_latitude'].between(40.7800, 40.7680)),'airport_lga'] = 1
    data_frame.loc[(data_frame['dropoff_longitude'].between(-73.8870, -73.8580)) & \
                   (data_frame['dropoff_latitude'].between(40.7800, 40.7680)), 'airport_lga'] = 1

    # Newark Liberty International Airport
    data_frame['airport_ewr'] = 0
    data_frame.loc[(data_frame['pickup_longitude'].between(-74.192, -74.172)) & \
                   (data_frame['pickup_latitude'].between(40.708, 40.676)),'airport_ewr'] = 1
    data_frame.loc[(data_frame['dropoff_longitude'].between(-74.192, -74.172)) & \
                   (data_frame['dropoff_latitude'].between(40.708, 40.676)), 'airport_ewr'] = 1

    data_frame.loc[(data_frame['airport_jfk'] == 1) & (data_frame['drop_flag'] == DROP_FLAG_DEFAULT), 'drop_flag'] = DROP_FLAG_SELECTED
    data_frame.loc[(data_frame['airport_lga'] == 1) & (data_frame['drop_flag'] == DROP_FLAG_DEFAULT), 'drop_flag'] = DROP_FLAG_SELECTED
    data_frame.loc[(data_frame['airport_ewr'] == 1) & (data_frame['drop_flag'] == DROP_FLAG_DEFAULT), 'drop_flag'] = DROP_FLAG_SELECTED

    return data_frame


def clean_invalid_records(data_frame):
    if 'fare_amount' in data_frame.columns:
        data_frame.loc[data_frame['fare_amount'] <= 0, 'drop_flag'] = DROP_FLAG_INVALID_FARE

    data_frame.loc[~data_frame['passenger_count'].between(1, 8),
                   'drop_flag'] = DROP_FLAG_INVALID_PASSENGER

    data_frame.loc[~data_frame['pickup_latitude'].between(-90, 90), 'drop_flag'] = DROP_FLAG_INVALID_VALUE
    data_frame.loc[~data_frame['pickup_longitude'].between(-180, 180), 'drop_flag'] = DROP_FLAG_INVALID_VALUE
    data_frame.loc[~data_frame['dropoff_latitude'].between(-90, 90), 'drop_flag'] = DROP_FLAG_INVALID_VALUE
    data_frame.loc[~data_frame['dropoff_longitude'].between(-180, 180), 'drop_flag'] = DROP_FLAG_INVALID_VALUE

    data_frame.loc[(data_frame['pickup_longitude'] == 0) & (data_frame['pickup_latitude'] == 0),
                   'drop_flag'] = DROP_FLAG_INVALID_LOCATION
    data_frame.loc[(data_frame['dropoff_longitude'] == 0) & (data_frame['dropoff_latitude'] == 0),
                   'drop_flag'] = DROP_FLAG_INVALID_LOCATION

    return data_frame


def calculate_distance(data_frame):
    data_frame['pickup_dropoff_distance'] = data_frame.apply(
        lambda row: haversine_distance(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']),
        axis=1)
    return data_frame


def haversine_distance(lat1, long1, lat2, long2):
    R = 6371  # radius of earth in kilometers
    # R = 3959 #radius of earth in miles
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(long2 - long1)

    # a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    # c = 2 * atan2( √a, √(1−a) )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # d = R*c
    d = (R * c)  # in kilometers
    return d


def add_drop_flag_column(data_frame):
    data_frame['drop_flag'] = DROP_FLAG_DEFAULT
    return data_frame


def set_weekend_flag(data_frame):
    data_frame['pickup_is_weekend'] = 0
    data_frame.loc[data_frame['pickup_weekday'].between(5, 6),
                   'pickup_is_weekend'] = 1
    return data_frame


def set_night_flag(data_frame):
    data_frame['pickup_is_night'] = 0
    data_frame.loc[data_frame['pickup_hour'].between(20, 23), 'pickup_is_night'] = 1
    data_frame.loc[data_frame['pickup_hour'].between(0, 6), 'pickup_is_night'] = 1
    return data_frame


def set_weekday_rush_hour_flag(data_frame):
    data_frame['pickup_is_rush_hour'] = 0
    data_frame.loc[(data_frame['pickup_weekday'].between(0, 4)) &
                   (data_frame['pickup_hour'].between(16, 19)), 'pickup_is_rush_hour'] = 1
    return data_frame


def set_order_cancelled_flag(data_frame):
    data_frame['is_order_cancelled'] = 0
    data_frame.loc[(data_frame['pickup_longitude'] != 0) & (data_frame['pickup_latitude'] != 0) &
                   (data_frame['pickup_longitude'] == data_frame['dropoff_longitude']) &
                   (data_frame['pickup_latitude'] == data_frame['dropoff_latitude']),
                    'is_order_cancelled'] = 1
    return data_frame


def clean_records_out_of_new_york(data_frame):
    outlier = ~data_frame['pickup_latitude'].between(37, 45)
    outlier |= ~data_frame['pickup_longitude'].between(-76, -69)
    outlier |= ~data_frame['dropoff_latitude'].between(37, 45)
    outlier |= ~data_frame['dropoff_longitude'].between(-76, -69)

    data_frame.loc[outlier & (data_frame['drop_flag'] == DROP_FLAG_DEFAULT), 'drop_flag'] = DROP_FLAG_INVALID_LOCATION

    return data_frame


if __name__ == '__main__':
    LOG_FORMAT = '[%(asctime)s] [%(lineno)d] [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--prj_dir',
                            type=str,
                            default='../')
    arg_parser.add_argument('--data_dir',
                            type=str,
                            default='data')
    arg_parser.add_argument('--raw_data_dir',
                            type=str,
                            default='raw-data')
    arg_parser.add_argument('--processing_data_dir',
                            type=str,
                            default='data-for-processing')
    arg_parser.add_argument('--input_data_formation',
                            type=str,
                            default='csv')
    arg_parser.add_argument('--output_data_formation',
                            type=str,
                            default='feather')
    FLAGS, _ = arg_parser.parse_known_args()

    start_time = dt.datetime.now()
    end_time = dt.datetime.now()

    data_dir = os.path.join(FLAGS.prj_dir, FLAGS.data_dir, FLAGS.processing_data_dir)
    if FLAGS.input_data_formation == 'feather':
        reg_exp = r'chunk.*\.feather|test.feather'
    else:
        reg_exp = r'chunk.*\.csv|test.csv'
    chunk_files = get_chunk_files(data_dir, reg_exp)
    logging.debug(chunk_files)
    # chunk_files=['chunk_011_train.feather', 'test.feather']
    # chunk_files=['test.feather']
    for file in chunk_files:
        file_path = os.path.join(data_dir, file)
        output_file_path = get_output_file_path(file_path, data_dir, FLAGS.output_data_formation)
        df = read_data_frame_from_file(file_path, FLAGS.input_data_formation)
        logging.debug('cleaning %s'%(file_path,))
        df = add_drop_flag_column(df)
        df = parse_date_time(df)
        df = set_weekend_flag(df)
        df = set_weekday_rush_hour_flag(df)
        df = set_night_flag(df)
        df = set_order_cancelled_flag(df)
        df = extract_airport_location(df)
        df = calculate_distance(df)
        if 'test' not in file:
            df = clean_invalid_records(df)
            df = clean_records_out_of_new_york(df)
        write_data_frame_to_file(df, output_file_path, FLAGS.output_data_formation)
        end_time = dt.datetime.now()
        logging.debug('done in %s'%(end_time-start_time,))
