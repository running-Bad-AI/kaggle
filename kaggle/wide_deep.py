# coding=UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import *
from tensorflow.python.estimator.export.export import build_parsing_serving_input_receiver_fn
import json
import os

from datetime import datetime



_BATCH_SIZE = 2

_CSV_COLUMNS = ['Name', 'AnimalType', 'AgeuponOutcome', 'HasName', 'Sterilized', 'Sex',
                'Year', 'Month', 'Day', 'Hour', 'IsMix', 'Breed_1', 'Breed_2',
                'Multiple_Breeds', 'Color_1', 'Color_2', 'Multiple_Colors', 'label']

_CSV_COLUMN_DEFAULTS = [[0]]*18

categorical_cols_buckets = {'Name': 7969,
                            'Hour': 24,  'Breed_1': 231, 'Breed_2': 159,
                            'Color_1': 57, 'Color_2': 47}

indicator_cols_buckets = {'AnimalType': 2, 'HasName': 2, 'Sterilized': 4, 'Sex': 3, 'Multiple_Colors': 2,
                          'Year': 4, 'Month': 12, 'Day': 7, 'IsMix': 2, 'Multiple_Breeds': 2, }

bucketized_cols_buckets = {'AgeuponOutcome': 8031,}

def initParser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_dir', type=str, default='./checkpoint',
        help='Base directory for the model.')

    parser.add_argument(
        '--model_type', type=str, default='wide_deep',
        help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

    parser.add_argument(
        '--train_epochs', type=int, default=50, help='Number of training epochs.')

    parser.add_argument(
        '--epochs_per_eval', type=int, default=1,
        help='The number of training epochs to run between evaluations.')

    parser.add_argument(
        '--batch_size', type=int, default=_BATCH_SIZE, help='Number of examples per batch.')

    parser.add_argument(
        '--train_data', type=str, default='./train_data/train',
        help='Path to the training data.')

    parser.add_argument(
        '--test_data', type=str, default='./train_data/validation',
        help='Path to the test data.')
    return parser


def build_model_columns():
    columns = {}
    categorical_columns = []
    embedding_columns = []
    indicator_columns = []
    bucketized_columns = []
    numeric_columns = []
    for col, bucket in categorical_cols_buckets.items():
        columns[col] = categorical_column_with_identity(col, num_buckets=categorical_cols_buckets[col], default_value=0)
        categorical_columns.append(columns[col])
        embedding_columns.append(embedding_column(columns[col], dimension=math.ceil(categorical_cols_buckets[col]**0.25)))

    for col, bucket in indicator_cols_buckets.items():
        columns[col] = categorical_column_with_identity(col, num_buckets=indicator_cols_buckets[col], default_value=0)
        categorical_columns.append(columns[col])
        indicator_columns.append(indicator_column(columns[col]))

    for col, bucket in bucketized_cols_buckets.items():
        columns[col] = numeric_column(col, dtype=tf.int64)
        numeric_columns.append(columns[col])
        bucketized_columns.append(bucketized_column(columns[col], boundaries=[7, 15, 30, 90, 180, 365, 2*365, 3*365, 5*365]))

    wide_columns = categorical_columns + bucketized_columns
    deep_columns = embedding_columns + indicator_columns + numeric_columns
    return wide_columns, deep_columns


def build_estimator(model_dir, model_type, wide_columns, deep_columns):
    """Build an estimator appropriate for the given model type."""
    hidden_units = [75, 50, 25]

    # opt = tf.train.AdamOptimizer()
    opt = tf.train.ProximalAdagradOptimizer(learning_rate=0.01, l1_regularization_strength=0.01,
                                            l2_regularization_strength=0.01)
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto())

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            n_classes=5,
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            n_classes=5,
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    elif model_type == 'wide_deep':
        return tf.estimator.DNNLinearCombinedClassifier(
            n_classes=5,
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.01,
                                                    l1_regularization_strength=0.01,
                                                    l2_regularization_strength=0.01,
                                                    ),
            dnn_optimizer=opt,
            config=run_config,
            dnn_dropout=0.5)



def input_fn(data_dir, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_dir), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_dir)

    print(data_dir)

    def parse_csv(value):
        print('Parsing', data_dir)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, field_delim=",")
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')

        return features, labels

    # Extract lines from input files using the Dataset API.
    data_files = [
        os.path.join(data_dir, f)
        for f in tf.gfile.ListDirectory(data_dir)
        if tf.gfile.Exists(os.path.join(data_dir, f))]
    dataset = tf.data.TextLineDataset(data_files)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=26728)

    dataset = dataset.map(parse_csv, num_parallel_calls=8)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def main(unused_argv):
    wide_columns, deep_columns = build_model_columns()
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type, wide_columns, deep_columns)

    def train_input_fn():
        return input_fn(
            FLAGS.train_data,
            100,
            True,
            FLAGS.batch_size)

    def eval_input_fn():
        return input_fn(
            FLAGS.test_data,
            1,
            False,
            FLAGS.batch_size)

    # res = model.predict(input_fn=eval_input_fn)
    # print(list(res)[0]["probabilities"])


    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

        results = model.evaluate(input_fn=lambda: input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


if __name__ == '__main__':

    start = datetime.now()
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = initParser()
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    end = datetime.now()
    print("time consume:%d sec" % (end - start).total_seconds())
