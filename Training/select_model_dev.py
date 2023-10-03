import os
import argparse
import csv 
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from utils import stats_report, get_metrics
from swa.tfkeras import SWA
from decision_b import dt_infer
import nni

SIZE=1250

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
    # sample = np.array(IEGM_seg, label)
    label = int(self.names_list[idx].split(' ')[1])
    sample = np.append(IEGM_seg, label)
    if self.append_path:
        sample = np.append(sample, text_path)
    # sample = {'IEGM_seg': IEGM_seg, 'label': label}
    return sample

# remain the model with v1 version
def model_features(input_size, params):
    if int(params['dense3']) != 0:
        model = keras.Sequential([
            keras.layers.Input(shape=input_size),
            keras.layers.Flatten(),
            keras.layers.Dropout(float(params['dropout1'])),
            keras.layers.Dense(int(params['dense1'])),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(float(params['dropout2'])),
            keras.layers.Dense(int(params['dense2'])),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(float(params['dropout3'])),
            keras.layers.Dense(int(params['dense3'])),
            keras.layers.ReLU(),
            keras.layers.Dense(2),
        ])
    elif int(params['dense2']) != 0:
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


def feature_extract(x: np.ndarray, nni_params, y=None, verbose=True):
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
            # features.append(factor_)
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

    features_list = features_list.reshape(features_list.shape[0], features_list.shape[1], 1)
    return features_list

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
    argparser.add_argument('--model', type=str, help="best or residual", 
                            default='model_features')
    argparser.add_argument('--enable_nni', type=str, help="enable nni for dt", 
                            default="False")
    argparser.add_argument('--model_params', type=str, help="model params with json format, provided if enable_nni=False", 
                            default="None")
        
    args = argparser.parse_args()
    return args

def run_once(args):
    # Hyperparameters
    LR = args.lr
    EPOCH = args.epoch
    SIZE = 1250
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
    if args.enable_nni == "True":
        params = nni.get_next_parameter()
    else:
        params = args.model_params
    x_aug = feature_extract(x_aug, params, y_aug)
    x_test = feature_extract(x_test, params, y_aug)
    LR = params['lr']
    EPOCH = int(params['epoch'])
    my_model = model_features(x_aug.shape[1:], params)
    import uuid
    import datetime
    save_name = f'{params["dense1"]}_{params["dense2"]}_{args.model}_{datetime.datetime.now()}'

    # save_name = 'SWA' 
    checkpoint_filepath = './20_10/' + save_name + '/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

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
    def get_metric(score_mode):
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
            x_test = feature_extract(x_test.numpy(), params)
            # x_test = np.expand_dims(x_test, axis=2)

            pred = my_model.predict(x_test).argmax(axis=1)
            segs_TP = 0
            segs_TN = 0
            segs_FP = 0
            segs_FN = 0

            for predicted_test, labels_test in zip(pred, y_test):
                if labels_test == 0:
                    segs_FP += int(predicted_test != labels_test)
                    segs_TN += int(predicted_test == labels_test)
                elif labels_test == 1:
                    segs_FN += int(predicted_test != labels_test)
                    segs_TP += int(predicted_test == labels_test)
            mylists.append([segs_TP, segs_FN, segs_FP, segs_TN])
        
        avgFB, G_score, detection_score = stats_report(mylists, f"{score_mode}_{save_name}")
        return avgFB, G_score, detection_score
    
    def get_score(score_mode="all"):
        if score_mode == "all":
            test_scores = get_score("test")
            train_score = get_score("train")
            test_scores.update(train_score)
            test_scores["default"] = train_score['trainScore']
            return test_scores
        else:
            avgFB, G_score, detection_score = get_metric(score_mode)
            return {
                f"{score_mode}Score": detection_score,
                f"{score_mode}avgFB": avgFB,
                f"{score_mode}G": G_score,
            }
    
    scores = get_score()
    nni.report_final_result(scores)

    return scores, my_model
    
def save_tf(path, model):
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   with tf.io.gfile.GFile(path, 'wb') as f:
      f.write(tflite_model)

if __name__ == '__main__':
    args = parse_args()
    run_once(args)