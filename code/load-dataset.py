# -*- coding=utf-8 -*-
import os
import pandas as pd
import logging
import argparse

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path',
                            type=str,
                            default='../data')
    arg_parser.add_argument('--input_dir',
                            type=str,
                            default='raw-data')
    arg_parser.add_argument('--output_dir',
                            type=str,
                            default='data-for-processing')
    arg_parser.add_argument('--train_data',
                            type=str,
                            default='train.csv')
    arg_parser.add_argument('--test_data',
                            type=str,
                            default='test.csv')
    FLAGS, _ = arg_parser.parse_known_args()

    train_csv_path = os.path.join(FLAGS.data_path, FLAGS.input_dir, FLAGS.train_data)
    test_csv_path = os.path.join(FLAGS.data_path, FLAGS.input_dir, FLAGS.test_data)

    train_org_path, _ = os.path.splitext(train_csv_path)
    test_org_path, _ = os.path.splitext(test_csv_path)

    train_feature_path = train_org_path + '.feather'
    test_feature_path = test_org_path + '.feather'

    columns_type = {
        'key': 'str',
        'fare_amount': 'float32',
        'pickup_datetime': 'str',
        'pickup_longitude': 'float32',
        'pickup_latitude': 'float32',
        'dropoff_longitude': 'float32',
        'dropoff_latitude': 'float32',
        'passenger_count': 'uint8'
    }
    columns_key = columns_type.keys()

    train_df = pd.read_csv(train_csv_path, usecols=columns_key, dtype=columns_type)
    logging.debug(train_df.head())
    logging.debug(train_df.info())
    train_df.to_feather(train_feature_path)
    logging.debug('%s done!' % (train_feature_path,))

    columns_type.pop('fare_amount')
    columns_key = columns_type.keys()

    test_df = pd.read_csv(test_csv_path, usecols=columns_key, dtype=columns_type)
    logging.debug(test_df.head())
    logging.debug(test_df.info())
    test_df.to_feather(test_feature_path)
    logging.debug('%s done!'%(test_feature_path,))