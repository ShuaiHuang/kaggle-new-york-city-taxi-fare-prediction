# -*- coding:utf-8 -*-
import os
import logging
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--prj_dir',
                            type=str,
                            default='../')
    arg_parser.add_argument('--data_dir',
                            type=str,
                            default='data')
    arg_parser.add_argument('--model_dir',
                            type=str,
                            default='model')
    arg_parser.add_argument('--train_data_dir',
                            type=str,
                            default='data-for-training')
    arg_parser.add_argument('--test_data',
                            type=str,
                            default='cleaned_test.feather')
    arg_parser.add_argument('--log_filename',
                            type=str,
                            default=None)
    arg_parser.add_argument('--model_md5',
                            type=str,
                            default='954debf1dd5ddfa5f94d06249ffcf398')
    FLAGS, _ = arg_parser.parse_known_args()

    train_data_dir = os.path.join(FLAGS.prj_dir, FLAGS.data_dir, FLAGS.train_data_dir)
    model_dir = os.path.join(FLAGS.prj_dir, FLAGS.model_dir)

    LOG_FORMAT = '[%(asctime)s] [%(lineno)d] [%(levelname)s] %(message)s'
    if FLAGS.log_filename is not None:
        LOG_FILE_PATH = os.path.join(train_data_dir, FLAGS.log_filename)
    else:
        LOG_FILE_PATH = None
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, filename=LOG_FILE_PATH)

    test_file_path = os.path.join(train_data_dir, FLAGS.test_data)
    test_df = pd.read_feather(test_file_path)
    test_df['fare_amount'] = np.nan
    test_buffer = xgb.DMatrix(test_df.drop(['key', 'fare_amount'], axis=1))

    logging.debug('Loading model...')
    model_filename = 'xgb_' + FLAGS.model_md5 + '.model'
    model_path = os.path.join(model_dir, model_filename)
    bst = xgb.Booster({'nthread': 12})
    bst.load_model(model_path)
    logging.debug('done!')

    logging.debug('Predicting test dataset...')
    prediction_result = bst.predict(test_buffer)
    test_df.loc[:, 'fare_amount'] = prediction_result
    logging.debug('done!')

    logging.debug('Saving submission file...')
    submission_filename = 'submission_' + FLAGS.model_md5 + '.csv'
    submission_path = os.path.join(train_data_dir, submission_filename)
    test_df = test_df.filter(items=['key', 'fare_amount'])
    test_df.to_csv(submission_path, index=False)
    logging.debug('%s done!', submission_filename)

