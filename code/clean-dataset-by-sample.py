# -*- coding:utf-8 -*-
import logging
import argparse
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from pytz import timezone


class DataCleaner(object):

    def __init__(self, data_dir, raw_data_dir, processing_data_dir, training_data_dir, train_data_path, test_data_path):
        self.__data_dir = data_dir
        self.__raw_data_dir = raw_data_dir
        self.__processing_data_dir = processing_data_dir
        self.__training_data_dir = training_data_dir
        self.__train_data_path = train_data_path
        self.__test_data_path = test_data_path

        self.__jfk_coord = {'longitude': (-73.778889 * np.pi) / 180, 'latitude': (40.639722 * np.pi) / 180}
        self.__ewr_coord = {'longitude': (-74.168611 * np.pi)/180, 'latitude': (40.6925 * np.pi)/180}
        self.__lga_coord = {'longitude': (-73.872611 * np.pi)/180, 'latitude': (40.77725 * np.pi)/180}
        self.__liberty_statue_coord = {'longitude': (-74.0445 * np.pi)/180, 'latitude': (40.6892 * np.pi)/180}
        self.__nyc_coord = {'longitude': (-74.0063889 * np.pi)/180, 'latitude': (40.7141667 * np.pi)/180}

        self.__radius_earth = 6371

    def load_dataset(self, nrows=10_000_000):
        train_stem = self.__train_data_path.stem
        converted_train_filename = train_stem + '.feather'
        converted_train_path = self.__processing_data_dir / converted_train_filename
        if converted_train_path.exists():
            logging.debug(f'load training dataset from {converted_train_path}')
            self.__train_df = pd.read_feather(converted_train_path, nthreads=6)
        else:
            logging.debug(f'load training dataset from {self.__train_data_path}')
            self.__train_df = pd.read_csv(self.__train_data_path, parse_dates=['pickup_datetime'], nrows=nrows)
            self.__train_df.to_feather(converted_train_path)
        logging.debug('done!')

        test_stem = self.__test_data_path.stem
        converted_test_filename = test_stem + '.feather'
        converted_test_path = self.__processing_data_dir / converted_test_filename
        if converted_test_path.exists():
            logging.debug(f'load testing dataset from {converted_test_path}')
            self.__test_df = pd.read_feather(converted_test_path, nthreads=6)
        else:
            logging.debug(f'load testing dataset from {self.__test_data_path}')
            self.__test_df = pd.read_csv(self.__test_data_path, parse_dates=['pickup_datetime'])
            self.__test_df.to_feather(converted_test_path)
        logging.debug('done!')

    def parse_datetime(self):
        self.__train_df = self.__extract_datetime_fields(self.__train_df)
        self.__test_df = self.__extract_datetime_fields(self.__test_df)

        self.__train_df['pickup_days_in_year'] = self.__train_df['pickup_datetime_local'].map(self.__extract_days_in_year)
        self.__train_df['pickup_days_sin'] = np.sin(2 * np.pi * self.__train_df['pickup_days_in_year'] / 365)
        self.__train_df['pickup_days_cos'] = np.cos(2 * np.pi * self.__train_df['pickup_days_in_year'] / 365)

        self.__test_df['pickup_days_in_year'] = self.__test_df['pickup_datetime_local'].map(self.__extract_days_in_year)
        self.__test_df['pickup_days_sin'] = np.sin(2 * np.pi * self.__test_df['pickup_days_in_year'] / 365)
        self.__test_df['pickup_days_cos'] = np.cos(2 * np.pi * self.__test_df['pickup_days_in_year'] / 365)

        self.__train_df['pickup_seconds_in_day'] = self.__extract_seconds_in_day(self.__train_df)
        self.__train_df['pickup_seconds_sin'] = np.sin(2 * np.pi * self.__train_df['pickup_seconds_in_day'] / 86400)
        self.__train_df['pickup_seconds_cos'] = np.cos(2 * np.pi * self.__train_df['pickup_seconds_in_day'] / 86400)

        self.__test_df['pickup_seconds_in_day'] = self.__extract_seconds_in_day(self.__test_df)
        self.__test_df['pickup_seconds_sin'] = np.sin(2 * np.pi * self.__test_df['pickup_seconds_in_day'] / 86400)
        self.__test_df['pickup_seconds_cos'] = np.cos(2 * np.pi * self.__test_df['pickup_seconds_in_day'] / 86400)

        self.__train_df['pickup_weekday_sin'] = np.sin(2 * np.pi * self.__train_df['pickup_weekday'] / 7)
        self.__train_df['pickup_weekday_cos'] = np.cos(2 * np.pi * self.__train_df['pickup_weekday'] / 7)

        self.__test_df['pickup_weekday_sin'] = np.sin(2 * np.pi * self.__test_df['pickup_weekday'] / 7)
        self.__test_df['pickup_weekday_cos'] = np.cos(2 * np.pi * self.__test_df['pickup_weekday'] / 7)

        self.__train_df = self.__get_time_class(self.__train_df)
        self.__test_df = self.__get_time_class(self.__test_df)

    def __get_time_class(self, data_frame):
        '''
        [0, 8): overnight - 0
        [8, 12): morning - 1
        [12, 16): afternoon - 2
        [16, 20): rush-hour - 3
        [20, 23]: night - 4
        :param data_frame:
        :return:
        '''
        data_frame['pickup_time_class'] = 0
        data_frame.loc[data_frame['pickup_hour'].between(0, 7), 'pickup_time_class'] = 0
        data_frame.loc[data_frame['pickup_hour'].between(8, 11), 'pickup_time_class'] = 1
        data_frame.loc[data_frame['pickup_hour'].between(12, 15), 'pickup_time_class'] = 2
        data_frame.loc[data_frame['pickup_hour'].between(16, 19), 'pickup_time_class'] = 3
        data_frame.loc[data_frame['pickup_hour'].between(20, 23), 'pickup_time_class'] = 4
        return data_frame

    def __extract_days_in_year(self, date):
        date_delta = date - dt.datetime(year=date.year, month=1, day=1, tzinfo=date.tzinfo)
        return date_delta.days

    def __extract_seconds_in_day(self, data_frame):
        return data_frame['pickup_hour'] * 3600 + data_frame['pickup_minute'] * 60 + data_frame['pickup_second']

    def __extract_datetime_fields(self, data_frame):
        data_frame = self.__convert_utc_to_us_eastern(data_frame)
        data_frame['pickup_year'] = data_frame['pickup_datetime_local'].map(lambda x: x.year)
        data_frame['pickup_month'] = data_frame['pickup_datetime_local'].map(lambda x: x.month)
        data_frame['pickup_day'] = data_frame['pickup_datetime_local'].map(lambda x: x.day)
        data_frame['pickup_hour'] = data_frame['pickup_datetime_local'].map(lambda x: x.hour)
        data_frame['pickup_minute'] = data_frame['pickup_datetime_local'].map(lambda x: x.minute)
        data_frame['pickup_second'] = data_frame['pickup_datetime_local'].map(lambda x: x.second)
        data_frame['pickup_weekday'] = data_frame['pickup_datetime_local'].map(lambda x: x.weekday())
        return data_frame

    def __convert_utc_to_us_eastern(self, data_frame):
        utc_timezone = timezone('utc')
        new_york_timezone = timezone('US/Eastern')
        data_frame['pickup_datetime_utc'] = data_frame['pickup_datetime'].map(
            lambda x: utc_timezone.localize(x))
        data_frame['pickup_datetime_local'] = data_frame['pickup_datetime_utc'].map(
            lambda x: x.astimezone(new_york_timezone))
        return data_frame

    def parse_coordinate(self):
        self.__train_df.loc[:, 'pickup_longitude'] = self.__convert_degree_to_raidus(
            self.__train_df['pickup_longitude'])
        self.__train_df.loc[:, 'pickup_latitude'] = self.__convert_degree_to_raidus(
            self.__train_df['pickup_latitude'])
        self.__train_df.loc[:, 'dropoff_longitude'] = self.__convert_degree_to_raidus(
            self.__train_df['dropoff_longitude'])
        self.__train_df.loc[:, 'dropoff_latitude'] = self.__convert_degree_to_raidus(
            self.__train_df['dropoff_latitude'])

        self.__test_df.loc[:, 'pickup_longitude'] = self.__convert_degree_to_raidus(
            self.__test_df['pickup_longitude'])
        self.__test_df.loc[:, 'pickup_latitude'] = self.__convert_degree_to_raidus(
            self.__test_df['pickup_latitude'])
        self.__test_df.loc[:, 'dropoff_longitude'] = self.__convert_degree_to_raidus(
            self.__test_df['dropoff_longitude'])
        self.__test_df.loc[:, 'dropoff_latitude'] = self.__convert_degree_to_raidus(
            self.__test_df['dropoff_latitude'])

        self.__train_df['longitude_delta'] = self.__train_df['dropoff_longitude'] - self.__train_df['pickup_longitude']
        self.__train_df['latitude_delta'] = self.__train_df['dropoff_latitude'] - self.__train_df['pickup_latitude']

        self.__test_df['longitude_delta'] = self.__test_df['dropoff_longitude'] - self.__test_df['pickup_longitude']
        self.__test_df['latitude_delta'] = self.__test_df['dropoff_latitude'] - self.__test_df['pickup_latitude']

        self.__train_df['haver_dist'] = self.__get_haversine_distance(self.__train_df)
        self.__test_df['haver_dist'] = self.__get_haversine_distance(self.__test_df)

        self.__train_df['bear_dist'] = self.__get_bearing_distance(self.__train_df)
        self.__test_df['bear_dist'] = self.__get_bearing_distance(self.__test_df)

        self.__train_df['jfk_dist'] = self.__train_df.apply(
            lambda x:
            self.__get_sphere_distance(
                x['pickup_latitude'],
                x['pickup_longitude'],
                self.__jfk_coord['latitude'],
                self.__jfk_coord['longitude']) +
            self.__get_sphere_distance(
                self.__jfk_coord['latitude'],
                self.__jfk_coord['longitude'],
                x['dropoff_latitude'],
                x['dropoff_longitude']),
            axis=1)
        self.__test_df['jfk_dist'] = self.__test_df.apply(
            lambda x:
            self.__get_sphere_distance(
                x['pickup_latitude'],
                x['pickup_longitude'],
                self.__jfk_coord['latitude'],
                self.__jfk_coord['longitude']) +
            self.__get_sphere_distance(
                self.__jfk_coord['latitude'],
                self.__jfk_coord['longitude'],
                x['dropoff_latitude'],
                x['dropoff_longitude']),
            axis=1)

        self.__train_df['ewr_dist'] = self.__train_df.apply(
            lambda x:
            self.__get_sphere_distance(
                x['pickup_latitude'],
                x['pickup_longitude'],
                self.__ewr_coord['latitude'],
                self.__ewr_coord['longitude']) +
            self.__get_sphere_distance(
                self.__ewr_coord['latitude'],
                self.__ewr_coord['longitude'],
                x['dropoff_latitude'],
                x['dropoff_longitude']),
            axis=1)
        self.__test_df['ewr_dist'] = self.__test_df.apply(
            lambda x:
            self.__get_sphere_distance(
                x['pickup_latitude'],
                x['pickup_longitude'],
                self.__ewr_coord['latitude'],
                self.__ewr_coord['longitude']) +
            self.__get_sphere_distance(
                self.__ewr_coord['latitude'],
                self.__ewr_coord['longitude'],
                x['dropoff_latitude'],
                x['dropoff_longitude']),
            axis=1)

        self.__train_df['lga_dist'] = self.__train_df.apply(
            lambda x:
            self.__get_sphere_distance(
                x['pickup_latitude'],
                x['pickup_longitude'],
                self.__lga_coord['latitude'],
                self.__lga_coord['longitude']) +
            self.__get_sphere_distance(
                self.__lga_coord['latitude'],
                self.__lga_coord['longitude'],
                x['dropoff_latitude'],
                x['dropoff_longitude']),
            axis=1)
        self.__test_df['lga_dist'] = self.__test_df.apply(
            lambda x:
            self.__get_sphere_distance(
                x['pickup_latitude'],
                x['pickup_longitude'],
                self.__lga_coord['latitude'],
                self.__lga_coord['longitude']) +
            self.__get_sphere_distance(
                self.__lga_coord['latitude'],
                self.__lga_coord['longitude'],
                x['dropoff_latitude'],
                x['dropoff_longitude']),
            axis=1)

        self.__train_df['liberty_dist'] = self.__train_df.apply(
            lambda x:
            self.__get_sphere_distance(
                x['pickup_latitude'],
                x['pickup_longitude'],
                self.__liberty_statue_coord['latitude'],
                self.__liberty_statue_coord['longitude']) +
            self.__get_sphere_distance(
                self.__liberty_statue_coord['latitude'],
                self.__liberty_statue_coord['longitude'],
                x['dropoff_latitude'],
                x['dropoff_longitude']),
            axis=1)
        self.__test_df['liberty_dist'] = self.__test_df.apply(
            lambda x:
            self.__get_sphere_distance(
                x['pickup_latitude'],
                x['pickup_longitude'],
                self.__liberty_statue_coord['latitude'],
                self.__liberty_statue_coord['longitude']) +
            self.__get_sphere_distance(
                self.__liberty_statue_coord['latitude'],
                self.__liberty_statue_coord['longitude'],
                x['dropoff_latitude'],
                x['dropoff_longitude']),
            axis=1)

        self.__train_df['nyc_dist'] = self.__train_df.apply(
            lambda x:
            self.__get_sphere_distance(
                x['pickup_latitude'],
                x['pickup_longitude'],
                self.__nyc_coord['latitude'],
                self.__nyc_coord['longitude']) +
            self.__get_sphere_distance(
                self.__nyc_coord['latitude'],
                self.__nyc_coord['longitude'],
                x['dropoff_latitude'],
                x['dropoff_longitude']),
            axis=1)
        self.__test_df['nyc_dist'] = self.__test_df.apply(
            lambda x:
            self.__get_sphere_distance(
                x['pickup_latitude'],
                x['pickup_longitude'],
                self.__nyc_coord['latitude'],
                self.__nyc_coord['longitude']) +
            self.__get_sphere_distance(
                self.__nyc_coord['latitude'],
                self.__nyc_coord['longitude'],
                x['dropoff_latitude'],
                x['dropoff_longitude']),
            axis=1)

    def __get_haversine_distance(self, data_frame):
        return 2 * self.__radius_earth * np.arcsin(np.sqrt(np.sin(data_frame['latitude_delta'] / 2) ** 2 + np.cos(data_frame['pickup_latitude']) * np.cos(data_frame['dropoff_latitude']) * np.sin(data_frame['longitude_delta'] / 2) ** 2))

    def __get_bearing_distance(self, data_frame):
        return np.arctan2(np.sin(-data_frame['longitude_delta'] * np.cos(data_frame['dropoff_latitude'])),
                          np.cos(data_frame['pickup_latitude'] * np.sin(data_frame['dropoff_latitude'])) - np.sin(data_frame['pickup_latitude']) * np.cos(data_frame['dropoff_latitude']) * np.cos(-data_frame['longitude_delta']))

    def __get_sphere_distance(self, pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
        lat_delta = dropoff_latitude - pickup_latitude
        lon_delta = dropoff_longitude - pickup_longitude

        return 2 * self.__radius_earth * np.arcsin(np.sqrt(np.sin(lat_delta / 2) ** 2 + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(lon_delta / 2) ** 2))

    def __convert_degree_to_raidus(self, series):
        return series / 180 * np.pi

    def clean_outlier(self):
        '''
        0 - reserved
        1 - fare_amount outlier
        2 - coordinate outlier
        3 - passenger_count outlier
        :return:
        '''
        self.__train_df['reserved_flag'] = 0

        self.__train_df.loc[~(self.__train_df['fare_amount'].between(2.5, 500)), 'reserved_flag'] = 1
        self.__train_df.loc[~(self.__train_df['pickup_latitude'].between(35, 45)), 'reserved_flag'] = 2
        self.__train_df.loc[~(self.__train_df['pickup_longitude'].between(-80, -70)), 'reserved_flag'] = 2
        self.__train_df.loc[~(self.__train_df['dropoff_latitude'].between(35, 45)), 'reserved_flag'] = 2
        self.__train_df.loc[~(self.__train_df['dropoff_longitude'].between(-80, -70)), 'reserved_flag'] = 2
        self.__train_df.loc[~(self.__train_df['passenger_count'].between(0, 9)), 'reserved_flag'] = 3

    def save_dataset(self):
        train_stem = self.__train_data_path.stem
        converted_train_filename = f'cleaned_{train_stem}.feather'
        converted_train_path = self.__processing_data_dir / converted_train_filename
        logging.debug(f'saving {converted_train_path}')
        self.__train_df.to_feather(converted_train_path)
        logging.debug('done!')

        test_stem = self.__test_data_path.stem
        converted_test_filename = f'cleaned_{test_stem}.feather'
        converted_test_path = self.__processing_data_dir / converted_test_filename
        logging.debug(f'saving {converted_test_path}')
        self.__test_df.to_feather(converted_test_path)
        logging.debug('done!')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--prj_dir',
                            type=str,
                            default='..')
    arg_parser.add_argument('--data_dir',
                            type=str,
                            default='data')
    arg_parser.add_argument('--raw_data',
                            type=str,
                            default='raw-data')
    arg_parser.add_argument('--data_for_processing',
                            type=str,
                            default='data-for-processing')
    arg_parser.add_argument('--data-for-training',
                            type=str,
                            default='data-for-training')
    arg_parser.add_argument('--train_data',
                            type=str,
                            default='train.csv')
    arg_parser.add_argument('--test_data',
                            type=str,
                            default='test.csv')
    arg_parser.add_argument('--log_file',
                            type=str,
                            default=None)
    FLAGS, _ = arg_parser.parse_known_args()

    PRJ_DIR = Path(FLAGS.prj_dir)
    DATA_DIR = PRJ_DIR / FLAGS.data_dir
    RAW_DATA_DIR = DATA_DIR / FLAGS.raw_data
    PROCESSING_DATA_DIR = DATA_DIR / FLAGS.data_for_processing
    TRAINING_DATA_DIR = DATA_DIR / FLAGS.data_for_training

    TRAIN_DATA_PATH = RAW_DATA_DIR / FLAGS.train_data
    TEST_DATA_PATH = RAW_DATA_DIR / FLAGS.test_data
    if FLAGS.log_file is None:
        LOG_FILE_PATH = None
    else:
        LOG_FILE_PATH = TRAINING_DATA_DIR / FLAGS.log_file

    LOG_FORMAT = '[%(asctime)s] [%(lineno)d] [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, filename=LOG_FILE_PATH)

    logging.debug(f'PRJ_DIR: {PRJ_DIR}')
    logging.debug(f'DATA_DIR: {DATA_DIR}')
    logging.debug(f'RAW_DATA_DIR: {RAW_DATA_DIR}')
    logging.debug(f'PROCESSING_DATA_DIR: {PROCESSING_DATA_DIR}')
    logging.debug(f'TRAINING_DATA_DIR: {TRAINING_DATA_DIR}')
    logging.debug(f'TRAIN_DATA_PATH: {TRAIN_DATA_PATH}')
    logging.debug(f'TEST_DATA_PATH: {TEST_DATA_PATH}')
    logging.debug(f'LOG_FILE_PATH: {LOG_FILE_PATH}')

    data_cleaner = DataCleaner(DATA_DIR, RAW_DATA_DIR, PROCESSING_DATA_DIR,
                               TRAINING_DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH)
    # data_cleaner.load_dataset(nrows=100)
    data_cleaner.load_dataset()
    data_cleaner.clean_outlier()
    data_cleaner.parse_datetime()
    data_cleaner.parse_coordinate()
    data_cleaner.save_dataset()
