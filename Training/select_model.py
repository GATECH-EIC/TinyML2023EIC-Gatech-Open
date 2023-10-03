import tempfile
import os
import argparse
import csv 
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import multiprocessing as mp
from utils import stats_report, get_metrics
from cosine_annealing import CosineAnnealingScheduler
from swa.tfkeras import SWA
from resnet import create_res_net, create_resnet18_best, transform_to_imgs, resnet152
from decision_b import dt_infer
import nni
from sklearn.preprocessing import StandardScaler

SIZE_MAP = {
    'none': 1250,
    'erase': 832,
    'average': 625,
    'drop_first': 625,
    'drop_last': 625,
    "remain_first": 1250 // 4,
    "remain_last": 1250 // 4,
}

# Define dataset
class IEGM_DataSET():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg, 'label': label}

        return sample

def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]

    return dictLabels

def get_subjects(csvf):
    """
    return all subjects id
    :param splitFile: csv file name
    """
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        ids = set()
        for i, row in enumerate(csvreader):
            ids.add(row[1].split('-')[0])
    return list(ids)

def txt_to_numpy(filename, row, merge_mode):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0
    if merge_mode == "none":
        for line in lines:
            line = line.strip().split(' ')
            datamat[row_count] = line[0]
            row_count += 1
    elif merge_mode == "average":
        for i in range(1, len(lines), 2):
            line_pre, line = lines[i-1].strip().split(' '), lines[i].strip().split(' ')
            datamat[row_count] = (float(line_pre[0]) + float(line[0])) / 2.
            row_count += 1
    elif merge_mode == "erase":
        for i in range(2, len(lines), 3):
            line_2, line_pre, line = lines[i-2].strip().split(' '), lines[i-1].strip().split(' '), lines[i].strip().split(' ')
            datamat[row_count] = (float(line_pre[0]) + float(line_2[0])) / 2.
            row_count += 1
            datamat[row_count] = (float(line_pre[0]) + float(line[0])) / 2.
            row_count += 1
    elif merge_mode == "drop_last":
        for i in range(len(lines) // 2):
            line = lines[i].strip().split(' ')
            datamat[row_count] = line[0]
            row_count += 1
    elif merge_mode == "drop_first":
        for i in range(len(lines) // 2, len(lines)):
            line = lines[i].strip().split(' ')
            datamat[row_count] = line[0]
            row_count += 1
    elif merge_mode == "remain_first":
        for i in range(0, len(lines) // 4):
            line = lines[i].strip().split(' ')
            datamat[row_count] = line[0]
            row_count += 1
    elif merge_mode == "remain_last":
        for i in range(len(lines) - (len(lines) // 4), len(lines)):
            line = lines[i].strip().split(' ')
            datamat[row_count] = line[0]
            row_count += 1
    else:
        assert True, "Unsupported merge mode"

    return datamat

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, root_dir, indice_dir, mode, size, merge_mode, subject_id=None,
               append_path=False):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.merge_mode = merge_mode
        self.append_path = append_path

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        # k=filename v=label
        for i, (k, v) in enumerate(csvdata_all.items()):
            if subject_id is None:
                self.names_list.append(str(k) + ' ' + str(v[0]))
            elif subject_id is not None and k.startswith(subject_id):
                self.names_list.append(str(k) + ' ' + str(v[0]))

  def __len__(self):
    return len(self.names_list)

  def __getitem__(self, idx):
    text_path = self.root_dir + self.names_list[idx].split(' ')[0]
    if not os.path.isfile(text_path):
      print(text_path + 'does not exist')
      return None

    IEGM_seg = txt_to_numpy(text_path, self.size, self.merge_mode).reshape(1, self.size, 1)
    label = int(self.names_list[idx].split(' ')[1])
    # sample = np.array(IEGM_seg, label)
    sample = np.append(IEGM_seg, label)
    if self.append_path:
        sample = np.append(sample, text_path)
    # sample = {'IEGM_seg': IEGM_seg, 'label': label}
    return sample

# Define the model architecture.
def model_best(merge_mode):
  input_size = (SIZE_MAP[merge_mode], 1)
  model = keras.Sequential([
      keras.layers.Input(shape=input_size),
      keras.layers.Conv1D(filters=3, kernel_size=85, strides=32, padding='valid', activation=None, use_bias=True),
      keras.layers.BatchNormalization(),
      keras.layers.ReLU(),
      keras.layers.Flatten(),

      keras.layers.Dropout(0.3),
      keras.layers.Dense(20),
      keras.layers.ReLU(),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(10),
      keras.layers.ReLU(),
      keras.layers.Dense(2),
  ])
  return model

def model_lstm(merge_mode):
  input_size = (SIZE_MAP[merge_mode], 1)
  model = keras.Sequential([
      keras.layers.Input(shape=input_size),
      keras.layers.LSTM(20, input_shape=input_size),
      keras.layers.Dense(2, activation='softmax')
  ])
  return model

def model_mix_lstm_conv(merge_mode):
    input_size = (SIZE_MAP[merge_mode], 1)
    model = keras.Sequential()

    model.add(keras.layers.Conv1D(32, 3, activation='relu', input_shape=input_size))
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.LSTM(20)) 

    model.add(keras.layers.Dense(2, activation='sigmoid'))
    return model

def model_convlstm(merge_mode):
    input_size = (SIZE_MAP[merge_mode], 1)
    model = keras.Sequential()
    model.add(keras.layers.Reshape((input_size[0], 1, 1), input_shape=input_size))
    model.add(keras.layers.ConvLSTM1D(filters=8, kernel_size=1, input_shape=input_size,
                                      activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    return model

def model_small_kernel(merge_mode):
      input_size = (SIZE_MAP[merge_mode], 1)
      model = keras.Sequential([
      keras.layers.Input(shape=input_size),
      keras.layers.Conv1D(filters=3, kernel_size=3, strides=1, padding='valid', activation=None, use_bias=True),
      keras.layers.BatchNormalization(),
      keras.layers.ReLU(),
      keras.layers.Flatten(),

      keras.layers.Dropout(0.3),
      keras.layers.Dense(20),
      keras.layers.ReLU(),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(10),
      keras.layers.ReLU(),
      keras.layers.Dense(2),
      ])
      return model

def model_resnet18(merge_mode):
    input_size = (67, 1)
    return create_res_net(input_size)

def model_resnet18best(merge_mode):
    input_size = (SIZE_MAP[merge_mode], 1)
    return create_resnet18_best(input_size)

def model_many_filter(merge_mode, filter_size=8):
    input_size = (SIZE_MAP[merge_mode], 1)
    model = keras.Sequential([
        keras.layers.Input(shape=input_size),
        keras.layers.Conv1D(filters=filter_size, kernel_size=85, strides=32, padding='valid', activation=None, use_bias=True),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Flatten(),

        keras.layers.Dropout(0.3),
        keras.layers.Dense(20),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(10),
        keras.layers.ReLU(),
        keras.layers.Dense(2),
    ])
    return model

def slight_manyfilter(merge_mode, filter_size=8):
    input_size = (SIZE_MAP[merge_mode], 1)
    model = keras.Sequential([
        keras.layers.Input(shape=input_size),
        keras.layers.Conv1D(filters=filter_size, kernel_size=85, strides=32, padding='valid', activation=None, use_bias=True),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Flatten(),

        keras.layers.Dropout(0.3),
        keras.layers.Dense(10),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(5),
        keras.layers.ReLU(),
        keras.layers.Dense(2),
    ])
    return model

def slight_manyfilter2(merge_mode, filter_size=8):
    input_size = (SIZE_MAP[merge_mode], 1)
    model = keras.Sequential([
        keras.layers.Input(shape=input_size),
        keras.layers.Conv1D(filters=filter_size, kernel_size=85, strides=32, padding='valid', activation=None, use_bias=True),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Flatten(),

        keras.layers.Dropout(0.3),
        keras.layers.Dense(14),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(7),
        keras.layers.ReLU(),
        keras.layers.Dense(2),
    ])
    return model

def model_many_filter2(merge_mode):
    input_size = (SIZE_MAP[merge_mode], 1)
    model = keras.Sequential([
        keras.layers.Input(shape=input_size),
        keras.layers.Conv1D(filters=16, kernel_size=85, strides=32, padding='valid', activation=None, use_bias=True),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Flatten(),

        keras.layers.Dropout(0.3),
        keras.layers.Dense(20),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(10),
        keras.layers.ReLU(),
        keras.layers.Dense(2),
    ])
    return model

def model_dense(merge_mode):
  input_size = (SIZE_MAP[merge_mode], 1)
  model = keras.Sequential([
      keras.layers.Input(shape=input_size),
      keras.layers.Conv1D(filters=3, kernel_size=85, strides=32, padding='valid', activation=None, use_bias=True),
      keras.layers.BatchNormalization(),
      keras.layers.ReLU(),
      keras.layers.Flatten(),

      keras.layers.Dropout(0.3),
      keras.layers.Dense(40),
      keras.layers.ReLU(),
      keras.layers.Dropout(0.3),
      keras.layers.Dense(20),
      keras.layers.ReLU(),
      keras.layers.Dropout(0.3),
      keras.layers.Dense(20),
      keras.layers.ReLU(),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(10),
      keras.layers.ReLU(),
      keras.layers.Dense(2),
  ])
  return model


def residual_model(merge_mode):
    input_size = (SIZE_MAP[merge_mode], 1)
    inputs = keras.Input(shape=input_size)
    x = keras.layers.Conv1D(filters=3, kernel_size=85, strides=32, padding='valid', activation=None, use_bias=True)(inputs)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.3)(x)

    x_small = keras.layers.Conv1D(filters=3, kernel_size=3, strides=32, padding='valid', activation=None, use_bias=True)(inputs)
    x_small = keras.layers.BatchNormalization()(x_small)
    x_small = keras.layers.ReLU()(x_small)
    x_small = keras.layers.Flatten()(x_small)
    x_small = keras.layers.Dropout(0.3)(x_small)
    
    x = keras.layers.Dense(20)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.1)(x)

    x_small = keras.layers.Dense(20)(x_small)
    x_small = keras.layers.ReLU()(x_small)
    x_small = keras.layers.Dropout(0.1)(x_small)

    # 为第一个Dense添加残差连接
    y = keras.layers.add([x, x_small]) 

    y = keras.layers.Dense(10)(y)
    y = keras.layers.ReLU()(y)
    outputs = keras.layers.Dense(2)(y)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def residual_model2(merge_mode):
    input_size = (SIZE_MAP[merge_mode], 1)
    inputs = keras.Input(shape=input_size)
    x = keras.layers.Conv1D(filters=8, kernel_size=85, strides=32, padding='valid', activation=None, use_bias=True)(inputs)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(20)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.1)(x)

    x_small = keras.layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='valid', activation=None, use_bias=True)(inputs)
    x_small = keras.layers.BatchNormalization()(x_small)
    x_small = keras.layers.ReLU()(x_small)
    x_small = keras.layers.Flatten()(x_small)
    x_small = keras.layers.Dropout(0.3)(x_small)

    x_small = keras.layers.Dense(20)(x_small)
    x_small = keras.layers.ReLU()(x_small)
    x_small = keras.layers.Dropout(0.1)(x_small)

    x_big = keras.layers.Conv1D(filters=8, kernel_size=200, strides=100, padding='valid', activation=None, use_bias=True)(inputs)
    x_big = keras.layers.BatchNormalization()(x_big)
    x_big = keras.layers.ReLU()(x_big)
    x_big = keras.layers.Flatten()(x_big)
    x_big = keras.layers.Dropout(0.3)(x_big)

    x_big = keras.layers.Dense(20)(x_big)
    x_big = keras.layers.ReLU()(x_big)
    x_big = keras.layers.Dropout(0.1)(x_big)

    # 为第一个Dense添加残差连接
    y = keras.layers.add([x, x_small, x_big]) 

    y = keras.layers.Dense(10)(y)
    y = keras.layers.ReLU()(y)
    outputs = keras.layers.Dense(2)(y)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def nni_model(merge_mode, params):
    input_size = (SIZE_MAP[merge_mode], 1)
    model = keras.Sequential([
        keras.layers.Input(shape=input_size),
        keras.layers.Conv1D(filters=int(params["filters"]), kernel_size=int(params["kernel_size"]), 
                            strides=int(params["strides"]), padding='valid', 
                            activation=None, use_bias=True),
        keras.layers.BatchNormalization(),
        keras.layers.Activation(params["activation"]),
        keras.layers.Flatten(),

        keras.layers.Dropout(params["dropout1"]),
        keras.layers.Dense(int(params["dense1"])),
        keras.layers.ReLU(),
        keras.layers.Dropout(params["dropout2"]),
        keras.layers.Dense(int(params["dense2"])),
        keras.layers.ReLU(),
        keras.layers.Dense(2),
    ])
    return model


def model_my(merge_mode, params):
    features = []
    input_size = (SIZE_MAP[merge_mode], 1)
    inputs = keras.layers.Input(input_size)
    for i in range(int(params['num_conv'])):
        x = keras.layers.Conv1D(filters=1, 
                                kernel_size=int(params[f'kernel_{i}']), 
                                strides=int(params[f'strides_{i}']), padding='valid', 
                                activation=None, use_bias=True)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        features.append(x)

    max_length = max([output.shape[1] for output in features])
    padded_features = [keras.layers.ZeroPadding1D(padding=(0, max_length - output.shape[1]))(output) for output in features]
    merged_features = keras.layers.concatenate(padded_features, axis=-1)

    x = keras.layers.Flatten()(merged_features)

    x = keras.layers.Dropout(float(params["dropout1"]))(x)
    x = keras.layers.Dense(int(params["dense1"]))(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(float(params["dropout2"]))(x)
    x = keras.layers.Dense(int(params["dense2"]))(x)
    x = keras.layers.ReLU()(x)
    outputs = keras.layers.Dense(2)(x)
    model = keras.Model(inputs, outputs)
    return model, float(params["lr"])


# remain the model with v1 version
def model_features(input_size, params=None):
    if params is None:
        params = {
            "dense1": 20,
            "dense2": 10,
            "dropout1": 0.3,
            "dropout2": 0.1
        }
    if int(params['dense2']) != 0:
        model = keras.Sequential([
            keras.layers.Input(shape=input_size),
            keras.layers.Flatten(),
            keras.layers.Dropout(float(params['dropout1'])),
            keras.layers.Dense(int(params['dense1'])),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(float(params['dropout2'])),
            keras.layers.Dense(int(params['dense2'])),
            keras.layers.ReLU(),
            keras.layers.Dense(2),
        ])
    else:
        model = keras.Sequential([
            keras.layers.Input(shape=input_size),
            keras.layers.Flatten(),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(float(params['dropout1'])),
            keras.layers.Dense(int(params['dense1'])),
            keras.layers.ReLU(),
            keras.layers.Dense(2),
        ])
    return model


# use different Conv1D respectively to capture different features
def nni_model2(merge_mode, params):
    features = []
    input_size = (SIZE_MAP[merge_mode], 1)
    inputs = keras.layers.Input(input_size)
    for i in range(int(params['num_conv'])):
        x = keras.layers.Conv1D(filters=1, 
                                kernel_size=int(params[f'kernel_{i}']), 
                                strides=int(params[f'strides_{i}']), padding='valid', 
                                activation=None, use_bias=True)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        features.append(x)

    max_length = max([output.shape[1] for output in features])
    padded_features = [keras.layers.ZeroPadding1D(padding=(0, max_length - output.shape[1]))(output) for output in features]
    merged_features = keras.layers.concatenate(padded_features, axis=-1)

    x = keras.layers.Flatten()(merged_features)

    x = keras.layers.Dropout(float(params["dropout1"]))(x)
    x = keras.layers.Dense(int(params["dense1"]))(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(float(params["dropout2"]))(x)
    x = keras.layers.Dense(int(params["dense2"]))(x)
    x = keras.layers.ReLU()(x)
    outputs = keras.layers.Dense(2)(x)
    model = keras.Model(inputs, outputs)
    return model


def feature_extract(x: np.ndarray, nni_params, enable_normalization=False, scaler=None, verbose=True):
    if nni_params is None:
        nni_params = {
                "factor_0": 1.9670153246602902,
                "factor_1": 1.52449440191099,
                "factor_2": 2.1325651457849966,
                "factor_3": 2.4630273044052347,
                "factor_4": 1.0517272141280138,
                "threshold_0": 9.1378526926664,
                "threshold_1": 10.190556111590308,
                "threshold_2": 10.055307656973925,
                "threshold_3": 10.129489863762178,
                "threshold_4": 9.09417302088325,
            }
    from decision_b import countPeaksNN, count_unpeak
    factor_size = 5
    factors = [nni_params[f'factor_{i}'] for i in range(factor_size)]
    thresholds = [nni_params[f'threshold_{i}'] for i in range(factor_size)]
    detect_gap = int(nni_params.get("detect_gap", 20))

    def selected_stat(idx, data):
        if len(idx) == 0:
            return 0, 0, 0, 0
        selected_values = [data[i] for i in idx]
        diff = np.diff(idx)
        if len(idx) == 1:
            return np.mean(selected_values), np.std(selected_values), 0, 0
        return np.mean(selected_values), np.std(selected_values), np.mean(diff), np.std(diff)

    features_list = []
    for idx, sample in enumerate(x):
        if verbose and idx % 100 == 0:
            print(f"\r{idx}/{len(x)}\r", end="")
        std = np.std(sample)
        mean = np.mean(sample)
        features = []
        for i in range(len(thresholds)):
            thresh=thresholds[i]
            factor_=factors[i]
            sample = sample.squeeze()
            numPeaks, peaksIdx = countPeaksNN(sample,factor_, std, detect_gap)
            numUnpeaks, unpeaksIdx = count_unpeak(sample, factor_, std, detect_gap)
            # numUnpeaks, repeated_unpeaks = count_unpeak(sample, factor_)
            features.append(factor_)
            features.append(numPeaks)
            features.append(int(numPeaks > thresh))
            features.append(numUnpeaks)
            features.append(int(numUnpeaks > thresh))
            features.extend(selected_stat(peaksIdx, sample))
            features.extend(selected_stat(unpeaksIdx, sample))
            # features.append(valid_SVt)
        features.append(std)
        features.append(mean)
        features = np.array(features, dtype=float)

        try:
            tf.debugging.check_numerics(tf.Variable(features), f"checking{idx}")
        except Exception as e:
            print(f"Nan in idx={idx}")
            print(features)
            input()
        features_list.append(features)
    
    if verbose:
        print()
    features_list = np.array(features_list).astype("float32")
    if enable_normalization:
        if scaler is None:
            scaler = StandardScaler()
            features_list = scaler.fit_transform(features_list)
        else:
            features_list = scaler.transform(features_list)
    features_list = features_list.reshape(features_list.shape[0], features_list.shape[1], 1)
    return features_list, scaler


NNI_MAP = {
    "0": nni_model,
    "1": nni_model2
}


def step_decay(step):
  initial_learning_rate = 0.0004
  decay_steps = 100
  alpha = 0.0001
  step = min(step, decay_steps)
  cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  return initial_learning_rate * decayed

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0002)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    argparser.add_argument('--merge_mode', type=str, help="merge data with (none, average, erase, drop_first, drop_last)", 
                            default='none')
    argparser.add_argument('--model', type=str, help="best or residual", 
                            default='best')
    argparser.add_argument('--nni_index', type=str, help="the model index you want to run for nni", 
                            default='0')
    argparser.add_argument('--parallel', type=str, help="True or False", 
                            default='True')
    argparser.add_argument('--filter_size', type=str, help="filter size for manyfilter model", 
                            default=8)
    argparser.add_argument('--test_model', type=str, help="input model path to test", 
                            default=None)
    argparser.add_argument('--decision_tree', type=str, help="if use decision tree", 
                            default="False")
    argparser.add_argument('--enable_nni', type=str, help="enable nni for dt", 
                            default="False")
    argparser.add_argument('--statistical', type=str, help="the stat method you use", 
                            default="False")
    argparser.add_argument('--model_params', type=str, help="model params with json format", 
                            default="None")
    argparser.add_argument('--check_all0', type=str, help="check all negative label subjects", 
                            default=None)
    argparser.add_argument('--enable_normalization', type=str, help="enable normalization", 
                            default=False)
        
    args = argparser.parse_args()
    return args

def run_once(count, args):
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = SIZE_MAP[args.merge_mode]
    MODEL = args.model
    path_data = args.path_data
    path_indices = args.path_indices
    # Data aug setting
    data_aug = True
    mix = False
    flip_peak = False
    flip_time = False
    add_noise = True

    train_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode='train', 
                                    size=SIZE, merge_mode=args.merge_mode)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_generator)
    train_dataset = train_dataset.shuffle(10).batch(len(train_generator))
    train_dataset = train_dataset.repeat()
    train_iterator = iter(train_dataset)

    test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode='test',
                                    size=SIZE, merge_mode=args.merge_mode)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    one_element = train_iterator.get_next()
    x, y = one_element[...,0:-1], one_element[...,-1]
    x = np.expand_dims(x, axis=2)

    test_samples = test_iterator.get_next()
    x_test, y_test = test_samples[...,0:-1], test_samples[...,-1]
    x_test = np.expand_dims(x_test, axis=2)
    
    if data_aug:
      if mix:
        x_aug, y_aug = np.concatenate((x, x_test), axis=0), np.concatenate((y, y_test), axis=0)
        print('Mix Data Finish!')    
      else:
        x_aug = np.copy(x)
        y_aug = np.copy(y)
        for i in range(len(x)):
          flip_p = random.random()
          flip_t = random.random()
          if flip_p < 0.5 and flip_peak:
            x_aug[i] = -x[i]
          if flip_t < 0.5 and flip_time:
            x_aug[i] = np.flip(x[i])
          if add_noise:
            max_peak = x_aug[i].max() * 0.05
            factor = random.random()
            # factor = 1
            noise = np.random.normal(0, factor * max_peak, (len(x_aug[i]), 1))
            x_aug[i] = x_aug[i] + noise

        print('flip Peak: ', flip_peak)
        print('Add Noise: ', add_noise) 
    
    start_epoch = 10
    swa = SWA(start_epoch=start_epoch, 
          lr_schedule='cyclic', 
          swa_lr=0.0001,
          swa_lr2=0.0005,
          swa_freq=5,
          batch_size=args.batchsz,
          verbose=1)

    print("Select Model: ", MODEL)
    if MODEL == "best":
        my_model = model_best(args.merge_mode)
    elif MODEL == "residual":
        my_model = residual_model(args.merge_mode)
    elif MODEL == "residual2":
        my_model = residual_model2(args.merge_mode)
    elif MODEL == "dense":
        my_model = model_dense(args.merge_mode)
    elif MODEL == "smallkernel":
        my_model = model_small_kernel(args.merge_mode)
    elif MODEL == "manyfilter":
        my_model = model_many_filter(args.merge_mode, args.filter_size)
    elif MODEL == "manyfilter2":
        my_model = model_many_filter2(args.merge_mode)
    elif MODEL == "resnet18":
        x_aug = feature_extract(x_aug, None)
        x_test = feature_extract(x_test, None)
        my_model = model_resnet18(args.merge_mode)
        params = None
    elif MODEL == "resnet18best":
        my_model = model_resnet18best(args.merge_mode)
    elif MODEL == "lstm":
        my_model = model_lstm(args.merge_mode)
    elif MODEL == "mix_lstm_conv":
        my_model = model_mix_lstm_conv(args.merge_mode)
    elif MODEL == "conv_lstm":
        my_model = model_convlstm(args.merge_mode)
    elif MODEL == "slight_manyfilter":
        my_model = slight_manyfilter(args.merge_mode, args.filter_size)
    elif MODEL == "slight_manyfilter2":
        my_model = slight_manyfilter2(args.merge_mode, args.filter_size)
    elif MODEL == "nni_model":
        params = nni.get_next_parameter()
        my_model = NNI_MAP[args.nni_index](args.merge_mode, params)
        LR = params['lr']
    elif MODEL == "model_my":
        my_model, LR = model_my(args.merge_mode, args.model_params)
    elif MODEL == "model_features":
        if args.enable_nni == "True":
            params = nni.get_next_parameter()
        else:
            params = args.model_params
        args.enable_normalization = bool(int(params.get("enable_normalization", 0)))
        x_aug, scaler = feature_extract(x_aug, params, args.enable_normalization)
        x_test, scaler = feature_extract(x_test, params, args.enable_normalization, scaler)
        LR = params['lr']
        EPOCH = int(params['epoch'])
        my_model = model_features(x_aug.shape[1:], params)
    elif MODEL == "model_resnet152":
        my_model = resnet152()
        x_aug = transform_to_imgs(x_aug)
        x_test = transform_to_imgs(x_test)
    else:
        assert True, f"Unsupported Model: {MODEL}"
    import uuid
    import datetime

    save_name = 'random_' + str(count) + "_" + args.model + "_" + str(uuid.uuid1()) 
    if args.model == "nni_model":
        save_name = f"nni_{nni.get_experiment_id()}_{nni.get_sequence_id()}_{nni.get_trial_id()}"
    elif args.model == "model_features":
        save_name = f'{params["dense1"]}_{params["dense2"]}_{args.model}_{datetime.datetime.now()}'
    elif args.model == "model_resnet152":
        save_name = f'resnet152_{datetime.datetime.now()}'
    # save_name = 'SWA' 
    checkpoint_filepath = './20_10/' + save_name + '/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Train the digit classification model
    # lrate = LearningRateScheduler(step_decay)
    lrate = CosineAnnealingScheduler(T_max=100, eta_max=4e-4, eta_min=2e-4)

    class AccuracyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if MODEL != "nni_model":
                return
            nni.report_intermediate_result(
                {
                    "val_accuracy": logs['val_accuracy'],
                    "loss": logs["loss"],
                    "accuracy": logs["accuracy"],
                    "epoch": epoch,
                    "default": float(logs['val_accuracy'])
                }
            )


    my_model.compile(optimizer=Adam(learning_rate=LR),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )
    if data_aug:
        if mix:
            my_model.fit(
                x_aug,
                y_aug,
                epochs=100,
                batch_size=args.batchsz,
                validation_split=0.3,
                shuffle=True,
                # validation_data=(x_test, y_test),
                callbacks=[model_checkpoint_callback]
            )
        else:
            my_model.fit(
                x_aug,
                y_aug,
                epochs=EPOCH,
                batch_size=args.batchsz,
                # validation_split=0.3,
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks=[model_checkpoint_callback, swa, AccuracyCallback()]
            )
    else:
        my_model.fit(
        x,
        y,
        epochs=100,
        batch_size=args.batchsz,
        # validation_split=0.1,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=[model_checkpoint_callback]
        )
    my_model.load_weights(checkpoint_filepath)
    score = my_model.evaluate(x_test, y_test)
    print('Model: ', save_name)
    print('acc', score[1])
    save_tf('./ckpt/' + save_name + '.tflite', my_model)

    # for each subject, create a data generator containing all segments in this subject
    def get_metric(score_mode, scaler):
        subjects = get_subjects(os.path.join(path_indices, f'{score_mode}_indice.csv'))
        mylists = []
        for subject in subjects:
            test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode=score_mode,
                                        size=SIZE, merge_mode=args.merge_mode, subject_id=subject)
            test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
            test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
            test_dataset = test_dataset.repeat()
            test_iterator = iter(test_dataset)

            test_samples = test_iterator.get_next()
            x_test, y_test = test_samples[...,0:-1], test_samples[...,-1]
            if MODEL == "model_resnet152":
                x_test = transform_to_imgs(x_test)
            elif MODEL == "model_features":
                x_test, scaler = feature_extract(x_test.numpy(), params, args.enable_normalization, scaler)
            # x_test = np.expand_dims(x_test, axis=2)

            pred = my_model.predict(x_test).argmax(axis=1)
            segs_TP = 0
            segs_TN = 0
            segs_FP = 0
            segs_FN = 0

            for predicted_test, labels_test in zip(pred, y_test.numpy()):
                if labels_test == 0:
                    segs_FP += (1 - (predicted_test == labels_test).sum()).item()
                    segs_TN += (predicted_test == labels_test).sum().item()
                elif labels_test == 1:
                    segs_FN += (1 - (predicted_test == labels_test).sum()).item()
                    segs_TP += (predicted_test == labels_test).sum().item()
            mylists.append([segs_TP, segs_FN, segs_FP, segs_TN])
        
        avgFB, G_score, detection_score = stats_report(mylists, f"{score_mode}_{save_name}")
        return avgFB, G_score, detection_score
    
    def get_score(score_mode="all", scaler=None):
        if score_mode == "all":
            test_scores = get_score("test")
            train_score = get_score("train")
            test_scores.update(train_score)
            test_scores["default"] = test_scores['testScore'] + train_score['trainScore']
            return test_scores
        else:
            avgFB, G_score, detection_score = get_metric(score_mode, scaler)
            return {
                f"{score_mode}Score": detection_score,
                f"{score_mode}avgFB": avgFB,
                f"{score_mode}G": G_score,
            }

    if not args.enable_normalization:
        scaler = None
    scores = get_score(scaler=scaler)
    nni.report_final_result(scores)

    return scores, my_model
    
def save_tf(path, model):
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   with tf.io.gfile.GFile(path, 'wb') as f:
      f.write(tflite_model)

'''
if __name__ == '__main__':
    args = parse_args()
    best_total = 0.0
    times = 1
    for i in range(2, times*5 + 1):
        if i // times == 0:
            args.merge_mode = 'erase'
        elif i // times == 1:
            args.merge_mode = 'average'
        elif i // times == 2:
            args.merge_mode = 'drop_first'
        elif i // times == 3:
            args.merge_mode = 'drop_last'
        elif i // times == 4:
            args.merge_mode = 'none'
        

        FB, G, total, my_model = run_once(i, args)
        if total > best_total:
            best_total = total
            save_tf('./20_10/best_' + str(i) + '_' + args.merge_mode + '.tflite', my_model)
            print('Current Best: ', best_total)
        print(total)
    print('Current Best: ', best_total)
'''


import multiprocessing

def run_and_save(i, args, params):
    args.model_params = params
    scores, my_model = run_once(i, args)
    return scores

def run_group(args, params, times):
    metrics = []
    for i in range(times):
        scores = run_and_save(i, args, params)
        metrics.append(scores)

    return params['dense1'], params["dense2"], metrics


def process_subject_for_test_model(subject, args, mode, SIZE, weight_path):
    interpreter = tf.lite.Interpreter(model_path=weight_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    SIZE = SIZE_MAP[args.merge_mode]

    wrong_dict = {}
    all_dict = {}
    test_generator = DataGenerator(root_dir=args.path_data, indice_dir=args.path_indices, mode=mode,
                                size=SIZE, merge_mode=args.merge_mode, subject_id=subject,
                                append_path=True)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    test_samples = test_iterator.get_next()
    x_test, y_test, data_paths = test_samples[...,0:-2], test_samples[...,-2], test_samples[...,-1]
    x_test = x_test.numpy().astype('float32')
    if str(weight_path).count("model_features") != 0:
        x_test = feature_extract(x_test, params, args.enable_normalization, None, verbose=False)
    
    pred = []
    for sample in x_test:
        sample = np.expand_dims(sample, 0)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        sample_pred = interpreter.get_tensor(output_details[0]['index']).argmax(axis=1)
        pred.append(sample_pred)
    pred = np.array(pred)
    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0

    for predicted_test, labels_test, path in zip(pred, y_test.numpy().astype('float32'), data_paths):
        category = str(path).split("/")[-1].split('-')[1]
        all_dict.setdefault(category, 0)
        all_dict[category] += 1
        if labels_test != predicted_test:
            wrong_dict.setdefault(category, 0)
            wrong_dict[category] += 1
        if labels_test == 0:
            segs_FP += (1 - (predicted_test == labels_test).sum()).item()
            segs_TN += (predicted_test == labels_test).sum().item()
        elif labels_test == 1:
            segs_FN += (1 - (predicted_test == labels_test).sum()).item()
            segs_TP += (predicted_test == labels_test).sum().item()
    mylists = [segs_TP, segs_FN, segs_FP, segs_TN]
    return mylists, wrong_dict, all_dict


def test_model(weight_path, args):
    SIZE = SIZE_MAP[args.merge_mode]
    mode = "test"

    # for each subject, create a data generator containing all segments in this subject
    subjects = get_subjects(os.path.join(args.path_indices, f'{mode}_indice.csv'))
    mylists = []
    wrong_dict = {}
    all_dict = {}
    
    pool = multiprocessing.Pool(processes=len(subjects))
    results = [pool.apply_async(process_subject_for_test_model, 
                                args=(subject, args, mode, SIZE, weight_path)) for subject in subjects]
    pool.close()
    pool.join()
    mylists = []
    from collections import Counter
    wrong_dict = Counter({})
    all_dict = Counter({})
    for result in results:
        mylist, wrongs, alls = result.get()
        mylists.append(mylist)
        wrong_dict = wrong_dict + Counter(wrongs)
        all_dict = all_dict + Counter(alls)
    wrong_dict = dict(wrong_dict)
    all_dict = dict(all_dict)
    
    save_name = f"test_{args.test_model.split('/')[-1]}_{args.merge_mode}"
    avgFB, G_score, detection_score = stats_report(mylists, save_name)

    for key in wrong_dict.keys():
        print(f"key[{key}], wrong num[{wrong_dict[key]}], propotion[{wrong_dict[key]/all_dict[key]}]")
    print(f"avgFB[{avgFB}] G[{G_score}] Score[{detection_score}]")

    return avgFB, G_score, detection_score



def process_subject(subject, args, mode, SIZE, nni_params):
    wrong_dict = {}
    all_dict = {}
    test_generator = DataGenerator(root_dir=args.path_data, indice_dir=args.path_indices, mode=mode,
                                size=SIZE, merge_mode=args.merge_mode, subject_id=subject,
                                append_path=True)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    test_samples = test_iterator.get_next()
    x_test, y_test, data_paths = test_samples[..., 0:-2], test_samples[..., -2], test_samples[..., -1]

    x_test = x_test.numpy().astype('float32')
    pred = dt_infer(x_test, data_paths, nni_params)

    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0

    for predicted_test, labels_test, path in zip(pred, y_test.numpy().astype('float32'), data_paths):
        category = str(path).split("/")[-1].split('-')[1]
        all_dict.setdefault(category, 0)
        all_dict[category] += 1
        if labels_test != predicted_test:
            wrong_dict.setdefault(category, 0)
            wrong_dict[category] += 1
        if labels_test == 0:
            segs_FP += (1 - (predicted_test == labels_test).sum()).item()
            segs_TN += (predicted_test == labels_test).sum().item()
        elif labels_test == 1:
            segs_FN += (1 - (predicted_test == labels_test).sum()).item()
            segs_TP += (predicted_test == labels_test).sum().item()
    mylists = [segs_TP, segs_FN, segs_FP, segs_TN]
    return mylists, wrong_dict, all_dict


def test_dt(args, enable_nni=False):
    SIZE = SIZE_MAP[args.merge_mode]

    # for each subject, create a data generator containing all segments in this subject
    mode = "train" if enable_nni else "test"
    mode = "train"
    subjects = get_subjects(os.path.join(args.path_indices, f'{mode}_indice.csv'))
    # subjects = ["S27"]
    if enable_nni:
        nni_params = nni.get_next_parameter()
    else:
        nni_params = {
            "SVT_bound_range": 2.7108682979449927,
            "SVT_length": 9,
            "SVT_lowerbound": 0.9381333803368032,
            "SVT_unpeak_lowerbound": 0.9942725313851941,
            "SVT_unpeak_upperbound": 1.147877561857916,
            "factor_0": 1.9670153246602902,
            "factor_1": 1.52449440191099,
            "factor_2": 2.1325651457849966,
            "factor_3": 2.4630273044052347,
            "factor_4": 1.0517272141280138,
            "svt_weight_0": 0.14272480710799051,
            "svt_weight_1": 0.8926404346670715,
            "svt_weight_2": 0.10898781872424924,
            "svt_weight_3": 0.07992866845819406,
            "svt_weight_4": 0.20273233681315594,
            "threshold_0": 9.1378526926664,
            "threshold_1": 10.190556111590308,
            "threshold_2": 10.055307656973925,
            "threshold_3": 10.129489863762178,
            "threshold_4": 9.09417302088325,
            "TRIAL_BUDGET": 3
        }

    pool = multiprocessing.Pool(processes=len(subjects))
    results = [pool.apply_async(process_subject, args=(subject, args, mode, SIZE, nni_params)) for subject in subjects]
    pool.close()
    pool.join()
    mylists = []
    from collections import Counter
    wrong_dict = Counter({})
    all_dict = Counter({})
    for result in results:
        mylist, wrongs, alls = result.get()
        mylists.append(mylist)
        wrong_dict = wrong_dict + Counter(wrongs)
        all_dict = all_dict + Counter(alls)
    wrong_dict = dict(wrong_dict)
    all_dict = dict(all_dict)
    
    save_name = f"test_dt"
    avgFB, G_score, detection_score = stats_report(mylists, save_name, subjects)
    propotion_dict = {}
    for key in wrong_dict.keys():
        print(f"key[{key}], wrong num[{wrong_dict[key]}], propotion[{wrong_dict[key]/all_dict[key]}]")
        propotion_dict[key] = wrong_dict[key]/all_dict[key]
    print(f"avgFB[{avgFB}] G[{G_score}] Score[{detection_score}]")

    if enable_nni:
        nni_result = {
            "default": detection_score,
            "avgFB": avgFB,
            "G": G_score
        }
        for category, wrong_p in propotion_dict.items():
            nni_result[f"{category}_wrong"] = f"{wrong_p*100}%"
        nni.report_final_result(nni_result)

    return avgFB, G_score, detection_score



def process_subject_original(subject, args, mode, SIZE):
    wrong_dict = {}
    all_dict = {}
    test_generator = DataGenerator(root_dir=args.path_data, indice_dir=args.path_indices, mode=mode,
                                size=SIZE, merge_mode=args.merge_mode, subject_id=subject,
                                append_path=True)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    test_samples = test_iterator.get_next()
    x_test, y_test, data_paths = test_samples[..., 0:-2], test_samples[..., -2], test_samples[..., -1]

    x_test = x_test.numpy().astype('float32')
    import decision_tree_b_o
    pred = decision_tree_b_o.dt_infer(x_test, data_paths)

    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0

    for predicted_test, labels_test, path in zip(pred, y_test.numpy().astype('float32'), data_paths):
        category = str(path).split("/")[-1].split('-')[1]
        all_dict.setdefault(category, 0)
        all_dict[category] += 1
        if labels_test != predicted_test:
            wrong_dict.setdefault(category, 0)
            wrong_dict[category] += 1
        if labels_test == 0:
            segs_FP += (1 - (predicted_test == labels_test).sum()).item()
            segs_TN += (predicted_test == labels_test).sum().item()
        elif labels_test == 1:
            segs_FN += (1 - (predicted_test == labels_test).sum()).item()
            segs_TP += (predicted_test == labels_test).sum().item()
    mylists = [segs_TP, segs_FN, segs_FP, segs_TN]
    return mylists, wrong_dict, all_dict



def test_dt_original(args):
    SIZE = SIZE_MAP[args.merge_mode]

    # for each subject, create a data generator containing all segments in this subject
    mode = "train"
    subjects = get_subjects(os.path.join(args.path_indices, f'{mode}_indice.csv'))
    # subjects = ["S27"]

    pool = multiprocessing.Pool(processes=len(subjects))
    results = [pool.apply_async(process_subject_original, args=(subject, args, mode, SIZE)) for subject in subjects]
    pool.close()
    pool.join()
    mylists = []
    from collections import Counter
    wrong_dict = Counter({})
    all_dict = Counter({})
    for result in results:
        mylist, wrongs, alls = result.get()
        mylists.append(mylist)
        wrong_dict = wrong_dict + Counter(wrongs)
        all_dict = all_dict + Counter(alls)
    wrong_dict = dict(wrong_dict)
    all_dict = dict(all_dict)
    
    save_name = f"test_dt"
    avgFB, G_score, detection_score = stats_report(mylists, save_name, subjects)
    propotion_dict = {}
    for key in wrong_dict.keys():
        print(f"key[{key}], wrong num[{wrong_dict[key]}], propotion[{wrong_dict[key]/all_dict[key]}]")
        propotion_dict[key] = wrong_dict[key]/all_dict[key]
    print(f"avgFB[{avgFB}] G[{G_score}] Score[{detection_score}]")

    return avgFB, G_score, detection_score


def test_dt_sklearn(args, enable_nni=False):
    SIZE = SIZE_MAP[args.merge_mode]

    mode = "train"
    #subjects = ["S27"]
    mylists = []
    wrong_dict = {}
    all_dict = {}

    train_generator = DataGenerator(root_dir=args.path_data, indice_dir=args.path_indices, mode=mode,
                                size=SIZE, merge_mode=args.merge_mode, subject_id=None,
                                append_path=False)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_generator)
    train_dataset = train_dataset.batch(len(train_generator))
    train_dataset = train_dataset.repeat()
    train_iterator = iter(train_dataset)

    train_samples = train_iterator.get_next()
    x_train, y_train = train_samples[...,0:-1], train_samples[...,-1]

    x_train = x_train.numpy().astype('float32')

    import decision_tree_sklearn as dts
    model = dts.get_model(x_train, y_train, args.statistical)

    mode = "test"
    subjects = get_subjects(os.path.join(args.path_indices, f'{mode}_indice.csv'))
    for subject in subjects:
        test_generator = DataGenerator(root_dir=args.path_data, indice_dir=args.path_indices, mode=mode,
                                    size=SIZE, merge_mode=args.merge_mode, subject_id=subject,
                                    append_path=True)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
        test_dataset = test_dataset.batch(len(test_generator))
        test_dataset = test_dataset.repeat()
        test_iterator = iter(test_dataset)

        test_samples = test_iterator.get_next()
        x_test, y_test, data_paths = test_samples[...,0:-2], test_samples[...,-2], test_samples[...,-1]

        x_test = x_test.numpy().astype('float32')
        pred = model.predict(x_test)

        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0

        for predicted_test, labels_test, path in zip(pred, y_test.numpy().astype('float32'), data_paths):
            category = str(path).split("/")[-1].split('-')[1]
            all_dict.setdefault(category, 0)
            all_dict[category] += 1
            if labels_test != predicted_test:
                wrong_dict.setdefault(category, 0)
                wrong_dict[category] += 1
            if labels_test == 0:
                segs_FP += (1 - (predicted_test == labels_test).sum()).item()
                segs_TN += (predicted_test == labels_test).sum().item()
            elif labels_test == 1:
                segs_FN += (1 - (predicted_test == labels_test).sum()).item()
                segs_TP += (predicted_test == labels_test).sum().item()
        mylists.append([segs_TP, segs_FN, segs_FP, segs_TN])
    
    save_name = f"test_dt"
    avgFB, G_score, detection_score = stats_report(mylists, save_name, subjects)
    propotion_dict = {}
    for key in wrong_dict.keys():
        print(f"key[{key}], wrong num[{wrong_dict[key]}], propotion[{wrong_dict[key]/all_dict[key]}]")
        propotion_dict[key] = wrong_dict[key]/all_dict[key]
    print(f"avgFB[{avgFB}] G[{G_score}] Score[{detection_score}]")

    if enable_nni:
        nni_result = {
            "default": detection_score,
            "avgFB": avgFB,
            "G": G_score
        }
        for category, wrong_p in propotion_dict.items():
            nni_result[f"{category}_wrong"] = f"{wrong_p*100}%"
        nni.report_final_result(nni_result)

    return avgFB, G_score, detection_score


PARAMS = [
    {
    "lr": 0.001835409597921791,
    "num_conv": 3,
    "kernel_0": 67,
    "strides_0": 16,
    "kernel_1": 60,
    "strides_1": 58,
    "kernel_2": 135,
    "strides_2": 58,
    "kernel_3": 137,
    "strides_3": 75,
    "dropout1": 0.2135324422047209,
    "dropout2": 0.15226297761502597,
    "dense1": 25,
    "dense2": 6
    },
    {
        "lr": 0.0002637802331938284,
        "num_conv": 4,
        "kernel_0": 42,
        "strides_0": 16,
        "kernel_1": 70,
        "strides_1": 60,
        "kernel_2": 144,
        "strides_2": 54,
        "kernel_3": 54,
        "strides_3": 87,
        "dropout1": 0.15250241241558554,
        "dropout2": 0.12806864951587857,
        "dense1": 21,
        "dense2": 5
    }
]

def all0_subjects(args):
    score_mode = args.check_all0
    SIZE = args.size
    subjects = get_subjects(os.path.join(args.path_indices, f'{score_mode}_indice.csv'))
    count_all0 = 0
    for idx, subject in enumerate(subjects):
        print(f"\rprocessing {idx+1}/{len(subjects)}", end="")
        test_generator = DataGenerator(root_dir=args.path_data, indice_dir=args.path_indices, mode=score_mode,
                                    size=SIZE, merge_mode=args.merge_mode, subject_id=subject)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
        test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
        test_dataset = test_dataset.repeat()
        test_iterator = iter(test_dataset)

        test_samples = test_iterator.get_next()
        x_test, y_test = test_samples[...,0:-1], test_samples[...,-1]
        y_test = y_test.numpy().astype(int)
        is_all0 = True
        for y in y_test:
            if y == 1:
                is_all0=False
                break
        if is_all0:
            count_all0 += 1
    print()
    print(f"All negative labels subject has {count_all0}, total {len(subjects)}, propotion{count_all0/len(subjects)}")


if __name__ == '__main__':
    args = parse_args()
    if args.check_all0 is not None:
        all0_subjects(args)
    elif args.statistical != "False":
        test_dt_sklearn(args)
    elif args.model == "nni_model":
        FB, G, det, my_model = run_once(0, args)
    elif args.test_model is not None:
        avgFB, G_score, detection_score = test_model(args.test_model, args)
    elif args.decision_tree == "True":
        enable_nni = (args.enable_nni == "True")
        test_dt(args, enable_nni)
    elif args.decision_tree == "Original":
        test_dt_original(args)
    elif args.parallel == "True":
        times = 1
        # merge_modes = ['erase', 'average', 'drop_first', 'drop_last', "remain_first", 'remain_last', 'none']
        # filter_sizes = [4, 5, 6, 7, 8, 9, 10]

        # 创建一个进程池来处理每个 merge_mode 分组
        best_FB_dict = {}

        import json

        with open("./nni_params/dt_nn.json") as file:
            PARAMS = json.load(file)

        with multiprocessing.Pool(len(PARAMS)) as pool:
            results = []

            # 启动每个进程来处理一个分组
            for params in PARAMS:
                result = pool.apply_async(run_group, (args, params, times))
                results.append(result)

            results = {(result.get()[0], result.get()[1]): result.get()[2] for result in results}

        with open("summary.csv", 'wb') as f:
            score_keys = results.items()[-1][-1].keys()
            csv_header = "dense1,dense2," + ",".join(score_keys) + "\n"
            f.write(csv_header.encode('utf-8'))
            for denses, metrics in results.items():
                for scores in metrics:
                    csv_values = f"{denses[0]},{denses[1]}"
                    for key in score_keys:
                        csv_values = csv_values + f",{scores[key]}"
                    f.write(f"{csv_values}\n".encode('utf-8'))
        print(results)
    elif args.parallel == "False":
        best_det = 0.0
        for i in range(1):
            scores, my_model = run_once(i, args)
            det = scores["default"]
            if det > best_det:
                best_det = det
                save_tf('./20_10/best_resnet' + str(i) + '.tflite', my_model)
                print('Current Best: ', best_det)
            print(det)
        print('Current Best: ', best_det)
