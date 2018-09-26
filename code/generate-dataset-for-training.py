# -*- coding:utf-8 -*-
import os
import re
import logging
import argparse
import pandas as pd


def get_cleaned_file_list(directory, input_is_feather):
    file_list = os.listdir(directory)
    data_file_list = []
    if input_is_feather:
        reg_exp = r'cleaned_.*\.feather'
    else:
        reg_exp = r'cleaed_.*\.csv'
    for file in file_list:
        if re.match(reg_exp, file) is not None:
            data_file_list.append(file)
    data_file_list.sort()
    return data_file_list


def get_data_frame(file_path, input_is_feather):
    if input_is_feather:
        df = pd.read_feather(file_path)
    else:
        df = pd.read_csv(file_path)
    return df


def select_feature_columns(data_frame):
    columns_to_keep = ['key', 'pickup_longitude', 'pickup_latitude',
                       'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
                       'pickup_year', 'pickup_days_sin',
                       'pickup_days_cos', 'pickup_seconds_sin', 'pickup_seconds_cos',
                       'pickup_weekday_sin', 'pickup_weekday_cos', 'pickup_time_class',
                       'haver_dist', 'bear_dist', 'mha_flag',
                       'jfk_dist', 'ewr_dist', 'lga_dist', 'liberty_dist', 'nyc_dist']
    if 'fare_amount'in data_frame.columns:
        columns_to_keep.append('fare_amount')
    data_frame = data_frame.filter(columns_to_keep)
    return data_frame


def drop_outlier_records(data_frame):
    data_frame = data_frame[data_frame['reserved_flag'] == 0]
    return data_frame


def process_dummy_feature(train_df, test_df):
    merged_df = pd.concat([train_df, test_df], ignore_index=True, copy=False, sort=False)

    dummy_columns = ['pickup_year', 'pickup_time_class']
    merged_df = pd.get_dummies(merged_df, columns=dummy_columns)
    train_df = merged_df[merged_df['key'].isin(train_df['key'])]
    test_df = merged_df[merged_df['key'].isin(test_df['key'])]
    return train_df, test_df


def write_data_frame(data_frame, file_path, output_is_feather):
    if output_is_feather:
        data_frame.reset_index(drop=True, inplace=True)
        data_frame.to_feather(file_path)
    else:
        data_frame.to_csv(file_path, index=False)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--prj_dir',
                            type=str,
                            default='../')
    arg_parser.add_argument('--data_dir',
                            type=str,
                            default='data')
    arg_parser.add_argument('--train_data_dir',
                            type=str,
                            default='data-for-training')
    arg_parser.add_argument('--process_data_dir',
                            type=str,
                            default='data-for-processing')
    arg_parser.add_argument('--input_data_formation',
                            type=str,
                            default='feather')
    arg_parser.add_argument('--output_data_formation',
                            type=str,
                            default='feather')
    arg_parser.add_argument('--log_filename',
                            type=str,
                            default=None)
    FLAGS, _ = arg_parser.parse_known_args()

    train_data_dir = os.path.join(FLAGS.prj_dir, FLAGS.data_dir, FLAGS.train_data_dir)
    process_data_dir = os.path.join(FLAGS.prj_dir, FLAGS.data_dir, FLAGS.process_data_dir)

    LOG_FORMAT = '[%(asctime)s] [%(lineno)d] [%(levelname)s] %(message)s'
    if FLAGS.log_filename is not None:
        LOG_FILE_PATH = os.path.join(process_data_dir, FLAGS.log_filename)
    else:
        LOG_FILE_PATH = None
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, filename=LOG_FILE_PATH)

    INPUT_FORMATION_IS_FEATHER = (FLAGS.input_data_formation == 'feather')
    OUTPUT_FORMATION_IS_FEATHER = (FLAGS.output_data_formation == 'feather')

    logging.debug('INPUT_FORMATION_IS_FEATHER=%s', INPUT_FORMATION_IS_FEATHER)
    logging.debug('OUTPUT_FORMATION_IS_FEATHER=%s', OUTPUT_FORMATION_IS_FEATHER)

    df_for_train = pd.read_feather(os.path.join(process_data_dir, 'cleaned_train.feather'))
    df_for_test = pd.read_feather(os.path.join(process_data_dir, 'cleaned_test.feather'))

    logging.debug(df_for_train.shape)
    logging.debug(df_for_train.columns)
    logging.debug(df_for_train.dtypes)

    df_for_train = drop_outlier_records(df_for_train)
    df_for_train = select_feature_columns(df_for_train)
    df_for_test = select_feature_columns(df_for_test)
    df_for_train, df_for_test = process_dummy_feature(df_for_train, df_for_test)

    output_file_path = os.path.join(train_data_dir, 'cleaned_train.feather')
    write_data_frame(df_for_train, output_file_path, OUTPUT_FORMATION_IS_FEATHER)
    logging.debug('save file: %s', output_file_path)

    output_file_path = os.path.join(train_data_dir, 'cleaned_test.feather')
    write_data_frame(df_for_test, output_file_path, OUTPUT_FORMATION_IS_FEATHER)
    logging.debug('save file: %s', output_file_path)

    logging.debug(df_for_train.shape)
    logging.debug(df_for_train.columns)
    logging.debug(df_for_test.shape)
    logging.debug(df_for_test.columns)