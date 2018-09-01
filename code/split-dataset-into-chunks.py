# -*- coding:utf-8 -*-
import os
import logging
import argparse
import pandas as pd
import datetime as dt


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
    arg_parser.add_argument('--output_data_formation',
                            type=str,
                            default='feather')
    arg_parser.add_argument('--chunk_size',
                            type=int,
                            default=5000000)
    FLAGS, _ = arg_parser.parse_known_args()

    raw_data_dir = os.path.join(FLAGS.data_path, FLAGS.input_dir)
    raw_data_path = os.path.join(FLAGS.data_path, FLAGS.input_dir, FLAGS.train_data)
    test_data_path = os.path.join(raw_data_dir, FLAGS.test_data)
    output_data_dir = os.path.join(FLAGS.data_path, FLAGS.output_dir)
    start_time = dt.datetime.now()

    csv_reader = pd.read_csv(raw_data_path, chunksize=FLAGS.chunk_size)
    for index, chunk_df in enumerate(csv_reader):
        if FLAGS.output_data_formation == 'feather':
            _, chunk_file_name = os.path.split(raw_data_path)
            chunk_file_name_base, _ = os.path.splitext(chunk_file_name)
            chunk_file_name = 'chunk_%03d_%s.feather'%(index, chunk_file_name_base)
        else:
            chunk_file_name = 'chunk_%03d_%s'%(index, FLAGS.train_data)

        chunk_file_path = os.path.join(output_data_dir, chunk_file_name)
        logging.debug('saving %s'%(chunk_file_name))

        if FLAGS.output_data_formation == 'feather':
            chunk_df = chunk_df.reset_index()
            chunk_df.to_feather(chunk_file_path)
        else:
            chunk_df.to_csv(chunk_file_path, index=False)
        end_time = dt.datetime.now()
        logging.debug('done in time %s'%(end_time-start_time,))

    if FLAGS.output_data_formation == 'feather':
        output_filename_base, _ = os.path.splitext(FLAGS.test_data)
        output_filename = '%s.feather'%(output_filename_base,)
        output_file_path = os.path.join(output_data_dir, output_filename)
        df = pd.read_csv(test_data_path)
        logging.debug('saving %s' % (output_filename))
        df.to_feather(output_file_path)
        end_time = dt.datetime.now()
        logging.debug('done in time %s' % (end_time - start_time,))
