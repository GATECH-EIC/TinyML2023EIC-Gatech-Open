import argparse
import serial
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os

def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float64)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat

def main():
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists('./log/'):
        os.makedirs('./log/')
    timeList = []
    port = args.com  # set port number
    ser = serial.Serial(port=port, baudrate=115200)  # set port number
    print(ser)
    ofp = open(file='log/res_{}.txt'.format(t), mode='w')  # make a new log file

    # Extract subject ID, filename, and label
    subject_data = {}
    with open('./test_indice.txt', 'r') as indice_file:
        for line in indice_file:
            label, filename = line.strip().split(',')
            subject_id = filename.split('-')[0]
            if subject_id not in subject_data:
                subject_data[subject_id] = []
            subject_data[subject_id].append((filename, label))

    all_metrics = []
    subjects_above_threshold = 0
    total_subjects = len(subject_data)

    # Perform calculations for each participant
    for subject_idx, (subject_id, file_info_list) in enumerate(subject_data.items(), start=1):
        y_true_subject = []
        y_pred_subject = []
        subject_desc = f'Subject {subject_id}:'
        file_tqdm = tqdm(file_info_list, desc=subject_desc, leave=True)

        fp = 0.0
        tn = 0.0
        fn = 0.0
        tp = 0.0

        for file_info in file_tqdm:
            filename, true_label = file_info
            y_true_subject.append(int(true_label))
            # load data from txt files and reshape to (1, 1, 1250, 1)
            testX = txt_to_numpy(args.path_data + filename, 1250).reshape(1, 1, 1250, 1)
            for i in range(0, testX.shape[0]):
                # don't continue running the code until a "begin" is received, otherwise receive iteratively
                while ser.in_waiting < 5:
                    pass
                    time.sleep(0.01)

                # when receiving the code "begin", send the test data cyclically
                recv = ser.read(size=ser.in_waiting).decode(encoding='utf8')
                # clear the input buffer
                ser.reset_input_buffer()
                if recv.strip() == 'begin':
                    for j in range(0, testX.shape[1]):
                        for k in range(0, testX.shape[2]):
                            for l in range(0, testX.shape[3]):
                                send_str = str(testX[i][j][k][l]) + ' '
                                ser.write(send_str.encode(encoding='utf8'))

                    # don't continue running the code until a "ok" is received
                    while ser.in_waiting < 2:
                        pass
                    time.sleep(0.01)
                    recv = ser.read(size=ser.in_waiting).decode(encoding='utf8')
                    ser.reset_input_buffer()
                    if recv.strip() == 'ok':
                        time.sleep(0.02)
                        # send status 200 to the board
                        send_str = '200 '
                        ser.write(send_str.encode(encoding='utf8'))
                        time.sleep(0.01)
                    # receive results from the board, which is a string separated by commas
                    while ser.in_waiting < 10:
                        pass
                    recv = ser.read(size=10).decode(encoding='utf8')
                    ser.reset_input_buffer()
                    # print(f"\n{recv}")
                    # the format of recv is ['<result>','<dutation>']
                    result = recv.split(',')[0]
                    inference_latency = recv.split(',')[1]
                    if result == '0':
                        y_pred_subject.append(0)
                    else:
                        y_pred_subject.append(1)
                    # inference latency in ms
                    timeList.append(float(inference_latency) * 1000)
                    ofp.write(str(result) + '\r')

        y_true_subject = np.array(y_true_subject)
        y_pred_subject = np.array(y_pred_subject)
        
        nva_idx = y_true_subject == 0
        va_idx = y_true_subject == 1
        
        fp += (len(y_true_subject[nva_idx]) - (y_pred_subject[nva_idx] == y_true_subject[nva_idx]).sum()).item()
        tn += ((y_pred_subject[nva_idx] == y_true_subject[nva_idx]).sum()).item()
        fn += (len(y_true_subject[va_idx]) - (y_pred_subject[va_idx] == y_true_subject[va_idx]).sum()).item()
        tp += ((y_pred_subject[va_idx] == y_true_subject[va_idx]).sum()).item()
        
        acc = (tn + tp) / (tn + fp + fn + tp)

        if (tp + fn == 0):
            precision = 1.0
        elif (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if (tp + fn) != 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 1.0

        if (fp + tn) != 0:
            FP_rate = fp / (fp + tn)
        else:
            FP_rate = 1.0

        # for the case: there is no VA segs for the patient
        if tp + fn == 0:
            PPV = 1
        # for the case: there is some VA segs
        elif tp + fp == 0 and tp + fn != 0:
            PPV = 0
        else:
            PPV = tp / (tp + fp)

        if (tn + fp) != 0:
            NPV = tn / (tn + fp)
        else:
            NPV = 1.0

        if (precision + sensitivity) != 0:
            F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
        else:
            F1_score = 0.0

        if ((2 ** 2) * precision + sensitivity) != 0:
            F_beta_score = (1 + 2 ** 2) * (precision * sensitivity) / ((2 ** 2) * precision + sensitivity)
        else:
            F_beta_score = 0.0
        
        all_metrics.append([acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score, F_beta_score])
        if F_beta_score > 0.95:
            subjects_above_threshold += 1

        # Update the progress bar
        if subject_idx < total_subjects:
            next_subject_id = list(subject_data.keys())[subject_idx]
            next_subject_desc = f'Files for Subject {next_subject_id}:'
            file_tqdm.set_description(next_subject_desc, refresh=True)
            file_tqdm.leave = True

    ofp.close()
    # Calculate average performance metrics
    total_time = sum(timeList)
    avg_time = np.mean(timeList)
    subject_metrics_array = np.array(all_metrics)
    average_metrics = np.mean(subject_metrics_array, axis=0)

    acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score, F_beta_score = average_metrics
    print("Final accuracy:", acc)
    print("Final precision:", precision)
    print("Final sensitivity:", sensitivity)
    print("Final FP_rate:", FP_rate)
    print("Final PPV:", PPV)
    print("Final NPV:", NPV)
    print("Final F1_score:", F1_score)
    print("Final F_beta_score:", F_beta_score)
    print("total_time:", total_time)
    print("avg_time:", avg_time)

    proportion_above_threshold = subjects_above_threshold / total_subjects
    g_score = proportion_above_threshold
    print("G Score:", g_score)

    f = open('./log/log_{}.txt'.format(t), 'a')
    f.write("Final Accuracy: {}\n".format(acc))
    f.write("Final Precision: {}\n".format(precision))
    f.write("Final Sensitivity: {}\n".format(sensitivity))
    f.write("Final FP_rate: {}\n".format(FP_rate))
    f.write("Final PPV: {}\n".format(PPV))
    f.write("Final NPV: {}\n".format(NPV))
    f.write("Final F1_Score: {}\n".format(F1_score))
    f.write("Final F_beta_Score: {}\n".format(F_beta_score))
    f.write("G score: {}\n\n".format(g_score))
    f.write("Total_Time: {}\n".format(total_time))
    f.write("Average_Time: {}\n\n".format(avg_time))
    f.close()



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--com', type=str, default='com15')
    argparser.add_argument('--path_data', type=str, default='F:/tinyml_contest_data_training/')
    args = argparser.parse_args()
    main()