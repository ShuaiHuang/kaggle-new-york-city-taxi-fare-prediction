# -*- coding:utf-8 -*-
import os
import re
import logging
import argparse
import pandas as pd
import datetime as dt

def get_chunk_files(path, reg_exp):
    file_list = os.listdir(path)
    for item in file_list:
        if not re.match(reg_exp, item):
            file_list.remove(item)
    file_list.sort()
    return file_list

def parse_date_time(file_path):
    chunk_df = pd.read_csv(file_path)
    chunk_df['pickup_timezone'] = chunk_df['pickup_datetime'].str.extract(r'\d+-\d+\-\d+\s\d+:\d+:\d+\s(.*)',
                                                                          expand=False)
    chunk_df['pickup_datetime'] = chunk_df['pickup_datetime'].map(
        lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S %Z"))
    chunk_df['pickup_year'] = chunk_df['pickup_datetime'].map(lambda x: x.year)
    chunk_df['pickup_month'] = chunk_df['pickup_datetime'].map(lambda x: x.month)
    chunk_df['pickup_day'] = chunk_df['pickup_datetime'].map(lambda x: x.day)
    chunk_df['pickup_hour'] = chunk_df['pickup_datetime'].map(lambda x: x.hour)
    chunk_df['pickup_minute'] = chunk_df['pickup_datetime'].map(lambda x: x.minute)
    chunk_df['pickup_second'] = chunk_df['pickup_datetime'].map(lambda x: x.second)
    chunk_df['pickup_weekday'] = chunk_df['pickup_datetime'].map(lambda x: x.weekday())
    chunk_df = chunk_df.drop(columns='pickup_datetime')
    path, filename = os.path.split(file_path)
    file_path = os.path.join(path, 'cleaned_'+filename)
    chunk_df.to_csv(file_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path',
                            type=str,
                            default='../data')
    arg_parser.add_argument('--input_dir',
                            type=str,
                            default='data-for-processing')
    arg_parser.add_argument('--test_file',
                            type=str,
                            default='test.csv')
    FLAGS, _ = arg_parser.parse_known_args()

    data_dir = os.path.join(FLAGS.data_path, FLAGS.input_dir)
    chunk_files = get_chunk_files(data_dir, r'chunk\_')
    for file in chunk_files:
        file_path = os.path.join(FLAGS.data_path, FLAGS.input_dir, file)
        logging.debug('cleaning %s'%(file_path,))
        parse_date_time(file_path)
        logging.debug('done!')
