# -*- coding:utf-8 -*-
import os
import re
import logging
import argparse
import numpy as np
import pandas as pd
import datetime as dt


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
        data_frame = data_frame.reset_index()
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
    data_frame['pickup_timezone'] = data_frame['pickup_datetime'].str.extract(r'\d+-\d+\-\d+\s\d+:\d+:\d+\s([A-Z]{3})', expand=False)
    data_frame['pickup_datetime_obj'] = data_frame['pickup_datetime'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))
    data_frame['pickup_year'] = data_frame['pickup_datetime_obj'].map(lambda x: x.year)
    data_frame['pickup_month'] = data_frame['pickup_datetime_obj'].map(lambda x: x.month)
    data_frame['pickup_day'] = data_frame['pickup_datetime_obj'].map(lambda x: x.day)
    data_frame['pickup_hour'] = data_frame['pickup_datetime_obj'].map(lambda x: x.hour)
    data_frame['pickup_minute'] = data_frame['pickup_datetime_obj'].map(lambda x: x.minute)
    data_frame['pickup_second'] = data_frame['pickup_datetime_obj'].map(lambda x: x.second)
    data_frame['pickup_weekday'] = data_frame['pickup_datetime_obj'].map(lambda x: x.weekday())
    return data_frame


def drop_records(data_frame):
    if 'fare_amount' in data_frame.columns:
        data_frame = data_frame.drop(data_frame[data_frame['fare_amount'] <= 0].index, axis=0)

    data_frame = data_frame.drop(data_frame[(data_frame['passenger_count'] <= 0) | (data_frame['passenger_count'] >= 9)].index, axis=0)

    data_frame = data_frame.drop(
        data_frame[(data_frame['pickup_latitude'] < -90) | (data_frame['pickup_latitude'] > 90)].index, axis=0)
    data_frame = data_frame.drop(
        data_frame[(data_frame['pickup_longitude'] < -180) | (data_frame['pickup_longitude'] > 180)].index, axis=0)
    data_frame = data_frame.drop(
        data_frame[(data_frame['dropoff_latitude'] < -90) | (data_frame['dropoff_latitude'] > 90)].index, axis=0)
    data_frame = data_frame.drop(
        data_frame[(data_frame['dropoff_longitude'] < -180) | (data_frame['dropoff_longitude'] > 180)].index, axis=0)

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
    for file in chunk_files:
        file_path = os.path.join(data_dir, file)
        output_file_path = get_output_file_path(file_path, data_dir, FLAGS.output_data_formation)
        df = read_data_frame_from_file(file_path, FLAGS.input_data_formation)
        logging.debug('cleaning %s'%(file_path,))
        df = parse_date_time(df)
        df = calculate_distance(df)
        if 'test' not in file:
            df = drop_records(df)
        write_data_frame_to_file(df, output_file_path, FLAGS.output_data_formation)
        end_time = dt.datetime.now()
        logging.debug('done in %s'%(end_time-start_time,))
