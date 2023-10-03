import argparse
import tensorflow as tf
import csv
import numpy as np
import os
from utils import stats_report
from sklearn.svm import SVC, LinearSVC

FeatureExtractParams = {
        "comment": "normalization is bad, add detect_gap, try use Evolution tunner",
        "lr": 0.00025400744543176943,
        "epoch": 63,
        "dropout1": 0.13163949420847665,
        "dropout2": 0.1697289465909091,
        "dense1": 16,
        "dense2": 6,
        "factor_0": 1.2576034326597791,
        "factor_1": 1.9181346667759376,
        "factor_2": 3.9560911940435046,
        "factor_3": 3.709846652603276,
        "factor_4": 2.143515823916947,
        "threshold_0": 7.557626730360952,
        "threshold_1": 10.535046907792262,
        "threshold_2": 8.90187206619138,
        "threshold_3": 7.050705405272044,
        "threshold_4": 9.73472882812338,
        "detect_gap": 18,
        "score": 194.17156
}

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

    IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
    label = int(self.names_list[idx].split(' ')[1])
    # sample = np.array(IEGM_seg, label)
    sample = np.append(IEGM_seg, label)
    if self.append_path:
        sample = np.append(sample, text_path)
    # sample = {'IEGM_seg': IEGM_seg, 'label': label}
    return sample
  

def feature_extract(x: np.ndarray, verbose=True):
    verbose = False

    nni_params = FeatureExtractParams

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
            print(f"\r{idx}/{len(x)}           ", end="")
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
        print("\r                  \r", end="")
    features_list = np.array(features_list).astype("float32")
    return features_list



def get_data(mode):
    SIZE = 1250
    path_indices = "./data_indices"
    path_data = "./tinyml_contest_data_training/"
    generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode=mode, 
                                    size=SIZE, subject_id="S01")
    dataset = tf.data.Dataset.from_tensor_slices(generator)
    dataset = dataset.shuffle(10).batch(len(generator))
    dataset = dataset.repeat()
    iterator = iter(dataset)

    one_element = iterator.get_next()
    x, y = one_element[...,0:-1], one_element[...,-1]
    x = np.expand_dims(x, axis=2)
    x = feature_extract(x)

    return x, y


def get_test(svc_classifier, pca):
    SIZE = 1250
    path_indices = "./data_indices"
    path_data = "./tinyml_contest_data_training/"

    mylists = []
    test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode="test",
                                size=SIZE, subject_id=None)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    test_samples = test_iterator.get_next()
    x_test, y_test = test_samples[...,0:-1], test_samples[...,-1]

    print("Start Feature Extr")
    x_test = feature_extract(x_test.numpy())
    # x_test = np.expand_dims(x_test, axis=2)

    print("Start PCA")
    x_test = pca.transform(x_test)

    pred = svc_classifier.predict(x_test)

    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0

    for predicted_test, labels_test in zip(pred, y_test.numpy().astype(int)):
        predicted_test = int(predicted_test)
        if labels_test == 0:
            segs_FP += (1 - (predicted_test == labels_test).sum()).item()
            segs_TN += (predicted_test == labels_test).sum().item()
        elif labels_test == 1:
            segs_FN += (1 - (predicted_test == labels_test).sum()).item()
            segs_TP += (predicted_test == labels_test).sum().item()
    mylists.append([segs_TP, segs_FN, segs_FP, segs_TN])
    
    avgFB, G_score, detection_score = stats_report(mylists, save_name=None)
    return avgFB, G_score, detection_score




def train(n_components):
    SIZE = 1250
    path_indices = "./data_indices"
    path_data = "./tinyml_contest_data_training/"

    mylists = []
    test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode="train",
                                size=SIZE, subject_id=None)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    test_samples = test_iterator.get_next()
    x_test, y_test = test_samples[...,0:-1], test_samples[...,-1]

    print("Start Feature Extr")
    x_test = feature_extract(x_test.numpy())
    # x_test = np.expand_dims(x_test, axis=2)

    print("Start SVC Training")

    from sklearn.decomposition import PCA
    print("Start PCA")
    pca = PCA(n_components=n_components)
    x_test = pca.fit_transform(x_test)

    svc_classifier = SVC()
    svc_classifier.fit(x_test, y_test.numpy())
    sv = svc_classifier.support_vectors_

    #coef_s, inter_s = svc_classifier.coef_.size, svc_classifier.intercept_.size
    #print(f"Parameter Size is {coef_s}+{inter_s}={coef_s+inter_s}")
    p_s = f"Parameter Size is {len(sv)}*{len(sv[0])}={len(sv)*len(sv[0])}"
    print(p_s)

    pred = svc_classifier.predict(x_test)

    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0

    for predicted_test, labels_test in zip(pred, y_test.numpy().astype(int)):
        predicted_test = int(predicted_test)
        if labels_test == 0:
            segs_FP += (1 - (predicted_test == labels_test).sum()).item()
            segs_TN += (predicted_test == labels_test).sum().item()
        elif labels_test == 1:
            segs_FN += (1 - (predicted_test == labels_test).sum()).item()
            segs_TP += (predicted_test == labels_test).sum().item()
    mylists.append([segs_TP, segs_FN, segs_FP, segs_TN])
    
    avgFB, G_score, detection_score = stats_report(mylists, save_name=None)
    testfb, testG, testScore = get_test(svc_classifier, pca)
    return avgFB, G_score, detection_score,testfb, testG, testScore, p_s




def get_metric(score_mode, interpreter:tf.lite.Interpreter):
    SIZE = 1250
    path_indices = "./data_indices"
    path_data = "./tinyml_contest_data_training/"
    subjects = get_subjects(os.path.join(path_indices, f'{score_mode}_indice.csv'))
    mylists = []
    input_details = interpreter.get_input_details()
    for subject in subjects:
        test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode=score_mode,
                                    size=SIZE, subject_id=subject)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
        test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
        test_dataset = test_dataset.repeat()
        test_iterator = iter(test_dataset)

        test_samples = test_iterator.get_next()
        x_test, y_test = test_samples[...,0:-1], test_samples[...,-1]

        x_test = feature_extract(x_test.numpy())
        # x_test = np.expand_dims(x_test, axis=2)

        pred = []
        for i in range(len(x_test)):
            input_data = np.expand_dims(x_test[i], axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred.append(output_data.argmax(axis=1))

        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0

        for predicted_test, labels_test in zip(pred, y_test.numpy().astype(int)):
            if labels_test == 0:
                segs_FP += (1 - (predicted_test == labels_test).sum()).item()
                segs_TN += (predicted_test == labels_test).sum().item()
            elif labels_test == 1:
                segs_FN += (1 - (predicted_test == labels_test).sum()).item()
                segs_TP += (predicted_test == labels_test).sum().item()
        mylists.append([segs_TP, segs_FN, segs_FP, segs_TN])
    
    avgFB, G_score, detection_score = stats_report(mylists, save_name=None)
    return avgFB, G_score, detection_score


def get_metric_by_weight(score_mode, dense1_w, dense1_b, dense2_w, dense2_b, dense3_w, dense3_b):
    SIZE = 1250
    path_indices = "./data_indices"
    path_data = "./tinyml_contest_data_training/"
    subjects = get_subjects(os.path.join(path_indices, f'{score_mode}_indice.csv'))
    mylists = []

    dense1_b = np.expand_dims(dense1_b, axis=1)
    dense2_b = np.expand_dims(dense2_b, axis=1)
    dense3_b = np.expand_dims(dense3_b, axis=1)

    dense1_w = dense1_w.astype("float32")
    dense1_b = dense1_b.astype("float32")
    dense2_w = dense2_w.astype("float32")
    dense2_b = dense2_b.astype("float32")
    dense3_w = dense3_w.astype("float32")
    dense3_b = dense3_b.astype("float32")
    '''
    print("{", end="")
    for j in range(2):
        print(f"{dense3_b[j][0]},",end="")
    print("}")
    print("{", end="")
    for i in range(6):
        for j in range(2):
            print(f"{dense3_w[j][i]},",end="")
    print("}")
    exit(0)
    '''

    for subject in subjects:
        if subject != "S48":
            continue
        test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode=score_mode,
                                    size=SIZE, subject_id=subject, append_path=True)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
        test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
        test_dataset = test_dataset.repeat()
        test_iterator = iter(test_dataset)

        test_samples = test_iterator.get_next()
        x_test, y_test, path = test_samples[...,0:-2], test_samples[...,-2], test_samples[...,-1]

        x_test = feature_extract(x_test.numpy().astype("float32"), path, False)
        # x_test = np.expand_dims(x_test, axis=2)

        pred = []
        for i in range(len(x_test)):
            input_data = x_test[i]
            input_data = input_data.astype("float32")
            input_data = np.matmul(dense1_w, input_data) + dense1_b # (16,62)*(62,1) + (16,1)
            input_data = np.maximum(0, input_data) # relu
            input_data = np.matmul(dense2_w, input_data) + dense2_b # (6,16)*(16,1) + (6,1)
            input_data = np.maximum(0, input_data) # relu
            input_data = np.matmul(dense3_w, input_data) + dense3_b # (2,6)*(6,1) + (2,1)
            input_data = np.argmax(input_data, axis=0)
            pred.append(input_data)

        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0

        for predicted_test, labels_test in zip(pred, y_test.numpy().astype(float).astype(int)):
            if labels_test == 0:
                segs_FP += (1 - (predicted_test == labels_test).sum()).item()
                segs_TN += (predicted_test == labels_test).sum().item()
            elif labels_test == 1:
                segs_FN += (1 - (predicted_test == labels_test).sum()).item()
                segs_TP += (predicted_test == labels_test).sum().item()
        mylists.append([segs_TP, segs_FN, segs_FP, segs_TN])
    
    avgFB, G_score, detection_score = stats_report(mylists, save_name=None)
    return avgFB, G_score, detection_score

with open("./train_result/svc.csv", "w") as f:
    results = {}
    for i in range(5,60):
        results[i] = train(n_components=i)
        print("n_c,fb,g,score,testFB,testG,testScore,p_size")
        print(results[i])
    print("Final Result is")
    for n_c, metric in results:
        print(f"{n_c}:  {metric}")
    f.write("n_c,fb,g,score,testFB,testG,testScore,p_size\n")
    for n_c, metric in results:
        f.write(f"{n_c},{metric[0]},{metric[1]},{metric[2]},{metric[3]},{metric[4]},{metric[5]},{metric[6]}")
