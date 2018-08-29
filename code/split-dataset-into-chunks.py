# -*- coding=utf-8 -*-
import os
import logging
import argparse
import pandas as pd

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
    arg_parser.add_argument('--chunk_size',
                            type=int,
                            default=5000000)
    FLAGS, _ = arg_parser.parse_known_args()

    raw_data_path = os.path.join(FLAGS.data_path, FLAGS.input_dir, FLAGS.train_data)

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

    csv_reader = pd.read_csv(raw_data_path, chunksize=FLAGS.chunk_size, usecols=columns_key, dtype=columns_type)
    for index, chunk_df in enumerate(csv_reader):
        chunk_file_name = 'chunk_%03d_%s'%(index, FLAGS.train_data)
        chunk_file_path = os.path.join(FLAGS.data_path, FLAGS.output_dir, chunk_file_name)
        logging.debug('saving %s'%(chunk_file_name))
        chunk_df.to_csv(chunk_file_path, index=False)
        logging.debug('done!')