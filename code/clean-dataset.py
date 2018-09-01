# -*- coding:utf-8 -*-
import os
import re
import logging
import argparse
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

def parse_date_time(input_file_path, output_file_dir=None):
    path, filename = os.path.split(input_file_path)
    if output_file_dir is None:
        cleaned_file_path = os.path.join(path, 'cleaned_' + filename)
    else:
        cleaned_file_path = os.path.join(output_file_dir, 'cleaned_' + filename)

    if not os.path.exists(cleaned_file_path):
        chunk_df = pd.read_csv(file_path)
        chunk_df['pickup_timezone'] = chunk_df['pickup_datetime'].str.extract(r'\d+-\d+\-\d+\s\d+:\d+:\d+\s([A-Z]{3})',
                                                                              expand=False)
        chunk_df['pickup_datetime'] = chunk_df['pickup_datetime'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))
        chunk_df['pickup_year'] = chunk_df['pickup_datetime'].map(lambda x: x.year)
        chunk_df['pickup_month'] = chunk_df['pickup_datetime'].map(lambda x: x.month)
        chunk_df['pickup_day'] = chunk_df['pickup_datetime'].map(lambda x: x.day)
        chunk_df['pickup_hour'] = chunk_df['pickup_datetime'].map(lambda x: x.hour)
        chunk_df['pickup_minute'] = chunk_df['pickup_datetime'].map(lambda x: x.minute)
        chunk_df['pickup_second'] = chunk_df['pickup_datetime'].map(lambda x: x.second)
        chunk_df['pickup_weekday'] = chunk_df['pickup_datetime'].map(lambda x: x.weekday())
        chunk_df = chunk_df.drop(columns='pickup_datetime')
        chunk_df.to_csv(cleaned_file_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
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
    arg_parser.add_argument('--test_file',
                            type=str,
                            default='test.csv')
    FLAGS, _ = arg_parser.parse_known_args()

    start_time = dt.datetime.now()
    end_time = dt.datetime.now()


    data_dir = os.path.join(FLAGS.prj_dir, FLAGS.data_dir, FLAGS.processing_data_dir)
    chunk_files = get_chunk_files(data_dir, 'chunk')
    logging.debug(chunk_files)
    for file in chunk_files:
        file_path = os.path.join(FLAGS.prj_dir, FLAGS.data_dir, FLAGS.processing_data_dir, file)
        logging.debug('cleaning %s'%(file_path,))
        parse_date_time(file_path)
        end_time = dt.datetime.now()
        logging.debug('done in %s'%(end_time-start_time,))

    test_file_path = os.path.join(FLAGS.prj_dir, FLAGS.data_dir, FLAGS.raw_data_dir, FLAGS.test_file)
    cleaned_test_file_dir = os.path.join(FLAGS.prj_dir, FLAGS.data_dir, FLAGS.processing_data_dir)
    logging.debug('cleaning %s'%(test_file_path,))
    parse_date_time(test_file_path, cleaned_test_file_dir)
    end_time = dt.datetime.now()
    logging.debug('done in %s' % (end_time - start_time,))
