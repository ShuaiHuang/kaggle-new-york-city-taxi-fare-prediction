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


def convert_data_formation(data_frame):
    columns_to_keep = ['key', 'pickup_longitude', 'pickup_latitude',
                       'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
                       'pickup_year', 'pickup_month', 'pickup_day',
                       'pickup_weekday', 'pickup_hour', 'pickup_dropoff_distance',
                       'airport_jfk', 'airport_lga', 'airport_ewr',
                       'pickup_is_weekend', 'pickup_is_night', 'pickup_is_rush_hour',
                       'is_order_cancelled']
    if 'fare_amount'in data_frame.columns:
        columns_to_keep.append('fare_amount')
    data_frame = data_frame.filter(columns_to_keep)

    dummy_columns = ['pickup_year', 'pickup_month', 'pickup_day', 'pickup_weekday', 'pickup_hour']
    data_frame = pd.get_dummies(data_frame, columns=dummy_columns)
    return data_frame


def write_data_frame(data_frame, file_path, output_is_feather):
    if output_is_feather:
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

    data_file_list = get_cleaned_file_list(process_data_dir, INPUT_FORMATION_IS_FEATHER)
    logging.debug(data_file_list)

    sampled_df_for_train = pd.DataFrame()
    df_for_test = None
    for filename in data_file_list:
        logging.debug('processing file: %s', filename)
        file_path = os.path.join(process_data_dir, filename)
        df = get_data_frame(file_path, INPUT_FORMATION_IS_FEATHER)
        if 'test' not in filename:
            reserved_df = df[df['drop_flag'] < 0]
            sampled_df = df[df['drop_flag'] == 0].sample(frac=0.1, random_state=43)
            sampled_df_for_train = pd.concat([sampled_df_for_train, reserved_df, sampled_df], ignore_index=True, copy=False)
        else:
            df_for_test = df

    # sampled_df_for_train.drop(index=sampled_df_for_train.loc[sampled_df_for_train['pickup_year']==2008, :].index, inplace=True)
    # sampled_df_for_train.reset_index(drop=True, inplace=True)

    sampled_df_for_train = convert_data_formation(sampled_df_for_train)
    output_file_path = os.path.join(train_data_dir, 'cleaned_train.feather')
    write_data_frame(sampled_df_for_train, output_file_path, OUTPUT_FORMATION_IS_FEATHER)
    logging.debug('save file: %s', output_file_path)

    df_for_test = convert_data_formation(df_for_test)
    output_file_path = os.path.join(train_data_dir, 'cleaned_test.feather')
    write_data_frame(df_for_test, output_file_path, OUTPUT_FORMATION_IS_FEATHER)
    logging.debug('save file: %s', output_file_path)

    logging.debug(sampled_df_for_train.shape)
    logging.debug(sampled_df_for_train.columns)
    logging.debug(df_for_test.shape)
    logging.debug(df_for_test.columns)