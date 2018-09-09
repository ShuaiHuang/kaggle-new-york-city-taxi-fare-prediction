# -*- coding:utf-8 -*-
import os
import re
import json
import shutil
import hashlib
import logging
import argparse
import pandas as pd
import xgboost as xgb
import datetime as dt
from sklearn.model_selection import train_test_split


def load_parameters_from_json(model_param_path):
    model_param_dir, model_param_filename = os.path.split(model_param_path)
    with open(model_param_path, 'r') as json_file:
        param = json.load(json_file)

    with open(model_param_path, 'rb') as json_file:
        json_md5_obj = hashlib.md5()
        json_md5_obj.update(json_file.read() + str(dt.datetime.now()).encode())
        json_md5_str = json_md5_obj.hexdigest()

    archive_json_filename = 'param_' + json_md5_str + '.json'
    if not os.path.exists(os.path.join(model_param_dir, archive_json_filename)):
        os.chdir(model_param_dir)
        shutil.copyfile(model_param_filename, archive_json_filename)
    return param, json_md5_str


def load_dataset_file(train_data_dir):
    file_list = os.listdir(train_data_dir)
    reg_exp = r'cleaned_train.feather'
    for file in file_list:
        if re.match(reg_exp, file) is not None:
            return file
    return None


def split_train_validate_dataset(data_file_list):
    train_file_list = data_file_list[:-2]
    validate_file_list = data_file_list[-2:]
    return train_file_list, validate_file_list


def write_history_to_file(model_dir, json_md5_str, result):
    log_file_path = os.path.join(model_dir, 'history.records')
    with open(log_file_path, 'a') as file_handler:
        content = '%s\t%s\t%f\n'%(dt.datetime.now(), json_md5_str, result)
        file_handler.write(content)


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
    arg_parser.add_argument('--log_filename',
                            type=str,
                            default=None)
    arg_parser.add_argument('--param_filename',
                            type=str,
                            default='param.json')
    FLAGS, _ = arg_parser.parse_known_args()

    train_data_dir = os.path.join(FLAGS.prj_dir, FLAGS.data_dir, FLAGS.train_data_dir)
    model_dir = os.path.join(FLAGS.prj_dir, FLAGS.model_dir)
    model_param_path = os.path.join(FLAGS.prj_dir, FLAGS.model_dir, FLAGS.param_filename)

    LOG_FORMAT = '[%(asctime)s] [%(lineno)d] [%(levelname)s] %(message)s'
    if FLAGS.log_filename is not None:
        LOG_FILE_PATH = os.path.join(train_data_dir, FLAGS.log_filename)
    else:
        LOG_FILE_PATH = None
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, filename=LOG_FILE_PATH)

    param, param_md5_str = load_parameters_from_json(model_param_path)

    data_file = load_dataset_file(train_data_dir)
    model_path = os.path.join(model_dir, 'xgb_'+param_md5_str+'.model')
    model_raw_path = os.path.join(model_dir, 'xgb_'+param_md5_str+'.model.raw.txt')
    logging.debug('data filename: %s', data_file)

    data_file_path = os.path.join(train_data_dir, data_file)
    data_df = pd.read_feather(data_file_path)

    train_df, validate_df = train_test_split(data_df, test_size=0.05, shuffle=True, random_state=param['model_param']['seed'])
    logging.debug(train_df.shape)
    logging.debug(validate_df.shape)

    train_data = xgb.DMatrix(train_df.drop(columns=['key', 'fare_amount'], axis=1), label=train_df['fare_amount'])

    validate_data = xgb.DMatrix(validate_df.drop(columns=['key', 'fare_amount'], axis=1),
                                label=validate_df['fare_amount'])
    eval_list = [(validate_data, 'eval')]
    num_round = param['num_round']
    early_stop_rounds = param['early_stop_rounds']
    progress_info = dict()

    logging.debug('training model...')
    xgb_model = xgb.train(param['model_param'],
                          train_data,
                          num_round,
                          eval_list,
                          early_stopping_rounds=early_stop_rounds,
                          evals_result=progress_info)
    logging.debug('done!')

    xgb_model.save_model(model_path)
    xgb_model.dump_model(model_raw_path)
    logging.debug('save model: %s', model_path)
    logging.debug('save model: %s', model_raw_path)
    write_history_to_file(model_dir, param_md5_str, progress_info['eval']['rmse'][-1])
