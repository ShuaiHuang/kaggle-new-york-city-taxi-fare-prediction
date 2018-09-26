# -*- coding: utf-8 -*-
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from sklearn.model_selection import train_test_split


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.debug('epoch: %d', epoch)


class DNNModel(object):
    def __init__(self, training_dataset_path, testing_dataset_path):
        logging.debug('loading training dataset and testing dataset...')
        self.__training_dataset_path = training_dataset_path
        self.__testing_dataset_path = testing_dataset_path

        raw_data_frame = pd.read_feather(self.__training_dataset_path)
        raw_data_label = raw_data_frame['fare_amount']
        raw_data_frame = raw_data_frame.drop(['fare_amount', 'key'], axis=1)
        self.__training_data_frame, self.__validation_data_frame, \
        self.__training_data_label, self.__validation_data_label = \
        train_test_split(raw_data_frame, raw_data_label, test_size=0.01)

        self.__testing_data_frame = pd.read_feather(self.__testing_dataset_path)
        self.__testing_data_frame = self.__testing_data_frame.drop(['fare_amount', 'key'], axis=1)
        logging.debug('done!')
        logging.debug(f'self.__training_data_frame.shape: {self.__training_data_frame.shape}')
        logging.debug(f'self.__training_data_label.shape: {self.__training_data_label.shape}')
        logging.debug(f'self.__validation_data_frame.shape: {self.__validation_data_frame.shape}')
        logging.debug(f'self.__validation_data_label.shape: {self.__validation_data_label.shape}')
        logging.debug(f'self.__testing_data_frame.shape: {self.__testing_data_frame.shape}')
        logging.debug(self.__training_data_frame.columns)

    def build_model(self):
        self.__model = keras.Sequential([
            keras.layers.Dense(72, activation='tanh', input_shape=(self.__training_data_frame.shape[1],)),
            keras.layers.Dense(72, activation='tanh'),
            keras.layers.Dense(36, activation='tanh'),
            keras.layers.Dense(36, activation='tanh'),
            keras.layers.Dense(18, activation='tanh'),
            keras.layers.Dense(1)
        ])

        optimizer = keras.optimizers.SGD(lr=0.1)

        self.__model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
        self.__model.summary()

    def train_model(self):
        mean = self.__training_data_frame.mean(axis=0)
        std = self.__training_data_frame.std(axis=0)
        self.__training_data_frame = (self.__training_data_frame - mean) / std
        self.__validation_data_frame = (self.__validation_data_frame - mean) /std
        self.__testing_data_frame = (self.__testing_data_frame - mean) / std

        self.__model.fit(self.__training_data_frame.values,
                         self.__training_data_label.values,
                         epochs=3,
                         validation_split=0.05,
                         callbacks=[PrintDot(),],
                         verbose=1,
                         batch_size=50)

        [loss, mse] = self.__model.evaluate(self.__validation_data_frame.values,
                                            self.__validation_data_label.values,
                                            verbose=1)
        logging.debug(f'loss={loss}, mse={mse}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--prj_dir',
                            type=str,
                            default='..')
    arg_parser.add_argument('--data_dir',
                            type=str,
                            default='data')
    arg_parser.add_argument('--training_data_dir',
                            type=str,
                            default='data-for-training')
    arg_parser.add_argument('--model_dir',
                            type=str,
                            default='model')
    arg_parser.add_argument('--training_dataset_filename',
                            type=str,
                            default='cleaned_train.feather')
    arg_parser.add_argument('--testing_dataset_filename',
                            type=str,
                            default='cleaned_test.feather')
    arg_parser.add_argument('--log_filename',
                            type=str,
                            default=None)
    FLAGS, _ = arg_parser.parse_known_args()

    PRJ_DIR = Path(FLAGS.prj_dir)
    DATA_DIR = PRJ_DIR / FLAGS.data_dir
    TRAINING_DATA_DIR = DATA_DIR / FLAGS.training_data_dir
    MODEL_DIR = PRJ_DIR / FLAGS.model_dir
    TRAINING_DATASET_PATH = TRAINING_DATA_DIR / FLAGS.training_dataset_filename
    TESTING_DATASET_PATH = TRAINING_DATA_DIR / FLAGS.testing_dataset_filename
    if FLAGS.log_filename is None:
        LOG_PATH = None
    else:
        LOG_PATH = TRAINING_DATA_DIR / FLAGS.log_filename

    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] [%(lineno)d] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, filename=LOG_PATH)

    logging.info(f'PRJ_DIR: {PRJ_DIR}')
    logging.info(f'DATA_DIR: {DATA_DIR}')
    logging.info(f'TRAINING_DATA_DIR: {TRAINING_DATA_DIR}')
    logging.info(f'MODEL_DIR: {MODEL_DIR}')
    logging.info(f'TRAINING_DATASET_PATH: {TRAINING_DATASET_PATH}')
    logging.info(f'TESTING_DATASET_PATH: {TESTING_DATASET_PATH}')
    logging.info(f'Tensorflow version: {tf.__version__}')

    dnn_model = DNNModel(TRAINING_DATASET_PATH, TESTING_DATASET_PATH)
    dnn_model.build_model()
    dnn_model.train_model()
