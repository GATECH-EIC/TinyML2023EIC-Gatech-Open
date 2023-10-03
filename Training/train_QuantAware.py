import os
import argparse
import csv 
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from utils import stats_report
from swa.tfkeras import SWA
import nni
import tensorflow_model_optimization as tfmot

result_dir = "quantV2"

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

def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0

    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, root_dir, indice_dir, mode, size, subject_id=None,
               append_path=False):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.append_path = append_path

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        # k=filename v=label
        for k, v in csvdata_all.items():
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

    IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
    label = int(self.names_list[idx].split(' ')[1])
    # sample = np.array(IEGM_seg, label)
    sample = np.append(IEGM_seg, label)
    if self.append_path:
        sample = np.append(sample, text_path)
    # sample = {'IEGM_seg': IEGM_seg, 'label': label}
    return sample

# remain the model with v1 version
def model_features(input_size, params, quant_firstlayer=False):
    if quant_firstlayer:
        dense1 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dense(int(params['dense1'])))
    else:
        dense1 = tf.keras.layers.Dense(int(params['dense1']))
    if int(params['dense2']) != 0:
        model = keras.Sequential([
            keras.layers.Input(shape=input_size),
            keras.layers.Flatten(),
            keras.layers.Dropout(float(params['dropout1'])),
            dense1,
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

def feature_extract(x: np.ndarray, nni_params, verbose=True):
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
    ensemble = 5
    factors = [nni_params[f'factor_{i}'] for i in range(ensemble)]
    thresholds = [nni_params[f'threshold_{i}'] for i in range(ensemble)]

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
        if verbose and idx % 1000 == 0:
            print(f"\r{idx}/{len(x)}\r", end="")
        std = np.std(sample)
        mean = np.mean(sample)
        features = []
        for i in range(len(thresholds)):
            thresh=thresholds[i]
            factor_=factors[i]
            sample = sample.squeeze()
            numPeaks, peaksIdx = countPeaksNN(sample,factor_, std)
            numUnpeaks, unpeaksIdx = count_unpeak(sample, factor_, std)
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
        print("\r Done")
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
    argparser.add_argument('--test_model', type=str, help="input model path to test", 
                            default=None)
    argparser.add_argument('--model_params', type=str, help="model params with json format", 
                            default="None")
    argparser.add_argument('--param_path', type=str, help="path of params", 
                            default=None)
    argparser.add_argument('--feature_extract', type=str, help="path of data to extract features", 
                            default=None)
    argparser.add_argument('--quant', type=str, help="quant opt: [INT8, FLOAT16]", 
                            default=None, choices=["INT8", "FLOAT16", "ALL"])
    argparser.add_argument('--quant_only_dense1', type=bool, help="quant only dense1 or all denses", 
                            default=False, choices=[True, False])
    
    args = argparser.parse_args()
    return args

def run_once(count, args):
    # Hyperparameters
    LR = args.lr
    EPOCH = args.epoch
    MODEL = args.model
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    params = args.model_params
    # Data aug setting
    data_aug = True
    mix = False
    flip_peak = False
    flip_time = False
    add_noise = True

    train_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode='train', 
                                    size=SIZE)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_generator)
    train_dataset = train_dataset.shuffle(10).batch(len(train_generator))
    train_dataset = train_dataset.repeat()
    train_iterator = iter(train_dataset)

    test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode='test',
                                    size=SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    one_element = train_iterator.get_next()

    x_aug, y_aug = one_element[...,0:-1], one_element[...,-1]
    x_aug = np.expand_dims(x_aug, axis=2)

    test_samples = test_iterator.get_next()
    x_test, y_test = test_samples[...,0:-1], test_samples[...,-1]
    x_test = np.expand_dims(x_test, axis=2)

    if data_aug:
      if mix:
        x_aug, y_aug = np.concatenate((x_aug, x_test), axis=0), np.concatenate((y_aug, y_test), axis=0)
        print('Mix Data Finish!')    
      else:
        x_aug = np.copy(x_aug)
        y_aug = y_aug.numpy()
        for i in range(len(x_aug)):
          flip_p = random.random()
          flip_t = random.random()
          if flip_peak and random.random() < 0.5 :
            x_aug[i] = -x_aug[i]
          if flip_time and random.random() < 0.5:
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
    x_aug = feature_extract(x_aug, params)
    x_test = feature_extract(x_test, params)
    LR = params['lr']
    EPOCH = int(params['epoch'])
    my_model = model_features(x_aug.shape[1:], params)
    
    import datetime
    save_name = f'{params["dense1"]}_{params["dense2"]}_{args.model}_'
    if not args.quant_only_dense1:
        save_name = save_name + "QAT"
    else:
        save_name = save_name + "Dense1"
    save_name = save_name.replace(" ", "-")
    checkpoint_filepath = './20_10/' + save_name + '/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Train the digit classification model
    # lrate = LearningRateScheduler(step_decay)
    # lrate = CosineAnnealingScheduler(T_max=100, eta_max=4e-4, eta_min=2e-4)

    my_model.compile(optimizer=Adam(learning_rate=LR),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )

    my_model.fit(
        x_aug,
        y_aug,
        epochs=EPOCH,
        batch_size=args.batchsz,
        # validation_split=0.3,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=[model_checkpoint_callback, swa]
    )

    my_model.load_weights(checkpoint_filepath)
    baseline_score = my_model.evaluate(x_test, y_test)
    print('Model: ', save_name)

    print("Start Quant Aware Training")
    if args.quant_only_dense1:
        def apply_quantization_to_dense(layer):
            if layer.name == "dense":
                return tfmot.quantization.keras.quantize_annotate_layer(layer)
            return layer
        my_model = tf.keras.models.clone_model(
            my_model,
            clone_function=apply_quantization_to_dense,
        )
        my_model = tfmot.quantization.keras.quantize_apply(my_model)
    else:
        quantize_model = tfmot.quantization.keras.quantize_model
        my_model = quantize_model(my_model)
    
    my_model.compile(optimizer=Adam(learning_rate=LR),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    my_model.summary()
    my_model.fit(
        x_aug,
        y_aug,
        epochs=EPOCH,
        batch_size=args.batchsz,
        # validation_split=0.3,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=[model_checkpoint_callback, swa]
    )

    quant_score = my_model.evaluate(x_test, y_test)
    print(f"Baseline Acc: {baseline_score}, Quant Acc: {quant_score}")

    save_tf(f'./train_ckpt/{result_dir}/{count}/' + save_name + '.tflite', my_model, None, x_aug)

    # for each subject, create a data generator containing all segments in this subject
    def get_metric(score_mode):
        subjects = get_subjects(os.path.join(path_indices, f'{score_mode}_indice.csv'))
        mylists = []
        for subject in subjects:
            test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode=score_mode,
                                        size=SIZE, subject_id=subject)
            test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
            test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
            test_dataset = test_dataset.repeat()
            test_iterator = iter(test_dataset)

            test_samples = test_iterator.get_next()
            x_test, y_test = test_samples[...,0:-1], test_samples[...,-1]
            x_test = feature_extract(x_test.numpy(), params)
            x_test = np.expand_dims(x_test, axis=2)

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
    
    def get_score(score_mode="all"):
        if score_mode == "all":
            test_scores = get_score("test")
            train_score = get_score("train")
            test_scores.update(train_score)
            return test_scores
        else:
            avgFB, G_score, detection_score = get_metric(score_mode)
            return {
                f"{score_mode}Score": detection_score,
                f"{score_mode}avgFB": avgFB,
                f"{score_mode}G": G_score,
                "default": detection_score
            }

    scores = get_score()
    nni.report_final_result(scores)

    return scores, my_model


def get_representative_data_gen(x_aug):
    def representative_data_gen():
        for input_value in x_aug:
            yield [input_value]
    return representative_data_gen


def save_tf(path, model, quant, x_aug):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quant is None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = get_representative_data_gen(x_aug)
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(path, 'wb') as f:
            f.write(tflite_model)
    elif quant == "INT8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = get_representative_data_gen(x_aug)
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(path, 'wb') as f:
            f.write(tflite_model)
    elif quant == "FLOAT16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(path, 'wb') as f:
            f.write(tflite_model)
    elif quant == "ALL":
        save_tf(f"{path}_INT8", model, "INT8", x_aug)
        save_tf(f"{path}_FLOAT16", model, "FLOAT16", x_aug)
        save_tf(f"{path}_NONE", model, None, x_aug)
    else:
        assert True, "Unsupported quant opt"

import multiprocessing
def run_and_save(i, args, params):
    args.model_params = params
    scores, _ = run_once(i, args)
    return scores

def run_group(i, args, params, times):
    metrics = []
    for _ in range(times):
        scores = run_and_save(i, args, params)
        metrics.append(scores)

    return params['dense1'], params["dense2"], metrics


def process_subject_for_test_model(subject, args, mode, SIZE, weight_path):
    params = {
        "lr": 0.01572549596010983,
        "epoch": 47,
        "dropout1": 0.10510658153349324,
        "dropout2": 0.20614333366441437,
        "dense1": 19,
        "dense2": 4,
        "factor_0": 1.583957402202563,
        "factor_1": 1.724842222010767,
        "factor_2": 2.322879493772209,
        "factor_3": 1.206815899492727,
        "factor_4": 2.8424094980222856,
        "threshold_0": 7.823403120088387,
        "threshold_1": 9.793277346441151,
        "threshold_2": 9.177149688173989,
        "threshold_3": 9.836094430363696,
        "threshold_4": 9.54879902045553
    }

    interpreter = tf.lite.Interpreter(model_path=weight_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    SIZE = args.size

    wrong_dict = {}
    all_dict = {}
    test_generator = DataGenerator(root_dir=args.path_data, indice_dir=args.path_indices, mode=mode,
                                size=SIZE, subject_id=subject,
                                append_path=True)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    test_samples = test_iterator.get_next()
    x_test, y_test, data_paths = test_samples[...,0:-2], test_samples[...,-2], test_samples[...,-1]
    x_test = x_test.numpy().astype('float32')
    if str(weight_path).count("model_features") != 0:
        x_test = feature_extract(x_test, params, verbose=False)
    
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
    SIZE = args.size
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
    
    save_name = f"test_{args.test_model.split('/')[-1]}"
    avgFB, G_score, detection_score = stats_report(mylists, save_name)

    for key in wrong_dict.keys():
        print(f"key[{key}], wrong num[{wrong_dict[key]}], propotion[{wrong_dict[key]/all_dict[key]}]")
    print(f"avgFB[{avgFB}] G[{G_score}] Score[{detection_score}]")

    return avgFB, G_score, detection_score


if __name__ == '__main__':
    args = parse_args()
    # test model, args.test_model is model weight tflite file
    if args.test_model is not None:
        avgFB, G_score, detection_score = test_model(args.test_model, args)
    # train model with hyper-params config, args.param_path is the path of hyper-params
    elif args.param_path is not None:
        for i in range(0, 50):
            times = 1
            best_FB_dict = {}
            import json
            with open(args.param_path) as file:
                PARAMS = json.load(file)
            PARAMS = [PARAMS[0]]
            with multiprocessing.Pool(len(PARAMS)) as pool:
                results = []

                for params in PARAMS:
                    result = pool.apply_async(run_group, (i, args, params, times))
                    results.append(result)

                results = {(result.get()[0], result.get()[1], PARAMS[idx]["score"]): result.get()[2] for idx, result in enumerate(results)}

            with open(f"./train_result/{result_dir}/summary{i}.csv", 'wb') as f:
                _, score_keys = list(results.items())[-1]
                score_keys = score_keys[0].keys()
                csv_header = "dense1,dense2,o_score," + ",".join(score_keys) + "\n"
                f.write(csv_header.encode('utf-8'))
                for idx, (denses, metrics) in enumerate(results.items()):
                    for scores in metrics:
                        csv_values = f"{denses[0]},{denses[1]},{denses[2]}"
                        for key in score_keys:
                            csv_values = csv_values + f",{scores[key]}"
                        f.write(f"{csv_values}\n".encode('utf-8'))
            print(results)
    # train model with below hyper-params
    elif args.feature_extract is not None:
        data = [txt_to_numpy(args.feature_extract, args.size)]
        param = {
            "lr": 0.01572549596010983,
            "epoch": 47,
            "dropout1": 0.10510658153349324,
            "dropout2": 0.20614333366441437,
            "dense1": 19,
            "dense2": 4,
            "factor_0": 1.583957402202563,
            "factor_1": 1.724842222010767,
            "factor_2": 2.322879493772209,
            "factor_3": 1.206815899492727,
            "factor_4": 2.8424094980222856,
            "threshold_0": 7.823403120088387,
            "threshold_1": 9.793277346441151,
            "threshold_2": 9.177149688173989,
            "threshold_3": 9.836094430363696,
            "threshold_4": 9.54879902045553
        }
        data = feature_extract(data, param, verbose=False)
        print(data)
    else:
        args.model_param = {
            "lr": 0.012944918126747171,
            "epoch": 51,
            "dropout1": 0.2209882186492696,
            "dropout2": 0.2189003154687614,
            "dense1": 14,
            "dense2": 11,
            "factor_0": 2.5770169300976833,
            "factor_1": 2.3031275006927774,
            "factor_2": 2.83169981838551,
            "factor_3": 2.581932582079781,
            "factor_4": 1.9231954900603407,
            "threshold_0": 9.815141312605427,
            "threshold_1": 9.30835053320244,
            "threshold_2": 10.146541932002439,
            "threshold_3": 7.128463465746124,
            "threshold_4": 7.40351351407921
        }
        best_det = 0.0
        times = 1
        for i in range(times):
            scores, my_model = run_once(args)
            det = scores["default"]
            if det > best_det:
                best_det = det
                print('Current Best: ', best_det)
            print(det)
        print('Current Best: ', best_det)
