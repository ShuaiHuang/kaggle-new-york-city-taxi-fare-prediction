# -*- coding:utf-8 -*-
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
                            default='data-for-processing')
    arg_parser.add_argument('--train_data',
                            type=str,
                            default='train.feather')
    arg_parser.add_argument('--test_data',
                            type=str,
                            default='test.feather')
    FLAGS, _ = arg_parser.parse_known_args()

    train_file_path = os.path.join(FLAGS.data_path, FLAGS.input_dir, FLAGS.train_data)
    logging.debug(train_file_path)
    train_df = pd.read_feather(train_file_path)
    logging.debug(train_df.info())