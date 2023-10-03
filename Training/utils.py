import csv, os
import numpy as np
from sklearn.metrics import confusion_matrix

def ACC(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc


def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]

    if tp + fn != 0:
        ppv = tp / (tp + fn)
    else:
        ppv = 0.
    return ppv


def Precision(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then ppv should be 1
    if tp + fn == 0:
        ppv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv

    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]

    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.
    return precision


def NPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then npv should be 1
    if tn + fp == 0:
        npv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn)
    return npv


def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity

    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:
        sensitivity = 0.0
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity


def Specificity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then spe should be 1
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity


def BAC(mylist):
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc


def F1(mylist):
    precision = Precision(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def FB(mylist, beta=2):
    precision = Precision(mylist)
    recall = Sensitivity(mylist)
    if ((beta**2)*precision + recall) == 0:
        f1 = 0
    else:
        f1 = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return f1

def get_metrics(mylist):
    f1 = F1(mylist)
    fb = FB(mylist)
    se = Sensitivity(mylist)
    sp = Specificity(mylist)
    bac = BAC(mylist)
    acc = ACC(mylist)
    ppv = PPV(mylist)
    npv = NPV(mylist)
    return fb

'''
def stats_report(mylist, save_name):
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    output = str(mylist) + '\n' + \
             "F-1 = " + str(f1) + '\n' + \
             "F-B = " + str(fb) + '\n' + \
             "SEN = " + str(se) + '\n' + \
             "SPE = " + str(sp) + '\n' + \
             "BAC = " + str(bac) + '\n' + \
             "ACC = " + str(acc) + '\n' + \
             "PPV = " + str(ppv) + '\n' + \
             "NPV = " + str(npv) + '\n'
    
    with open("./result/" + str(save_name) + ".txt", "w") as f:
        f.write(output)

    print("F-1 = ", F1(mylist))
    print("F-B = ", FB(mylist))
    print("SEN = ", Sensitivity(mylist))
    print("SPE = ", Specificity(mylist))
    print("BAC = ", BAC(mylist))
    print("ACC = ", ACC(mylist))
    print("PPV = ", PPV(mylist))
    print("NPV = ", NPV(mylist))
    return FB(mylist)
'''

def stats_report(mylists, save_name, subjects=[]):
    f1, fb, se, sp, bac, acc, ppv, npv = 0., 0., 0., 0., 0., 0., 0., 0.
    subject_above_threshold = 0
    fb_list = []
    for mylist in mylists:
        f1 += round(F1(mylist), 5)
        cur_fb = FB(mylist)
        fb_list.append(cur_fb)
        fb += round(cur_fb, 5)
        se += round(Sensitivity(mylist), 5)
        sp += round(Specificity(mylist), 5)
        bac += round(BAC(mylist), 5)
        acc += round(ACC(mylist), 5)
        ppv += round(PPV(mylist), 5)
        npv += round(NPV(mylist), 5)
        if cur_fb > 0.95:
            subject_above_threshold += 1
    f1 /= len(mylists)
    fb /= len(mylists)
    se /= len(mylists)
    sp /= len(mylists)
    bac /= len(mylists)
    acc /= len(mylists)
    ppv /= len(mylists)
    npv /= len(mylists)

    g_score = subject_above_threshold / len(mylists)
    detection_score = 70 * fb + 30 * g_score

    output = str(mylists) + '\n' + \
        "F-1 = " + str(f1) + '\n' + \
        "F-B = " + str(fb) + '\n' + \
        "G   = " + str(g_score) + '\n' + \
        "Det = " + str(detection_score) + '\n' + \
        "FBs = " + str(fb_list) + '\n' + \
        'Sub = ' + str(subjects) + '\n' + \
        "SEN = " + str(se) + '\n' + \
        "SPE = " + str(sp) + '\n' + \
        "BAC = " + str(bac) + '\n' + \
        "ACC = " + str(acc) + '\n' + \
        "PPV = " + str(ppv) + '\n' + \
        "NPV = " + str(npv) + '\n'
    
    if save_name is not None:
        with open("./result/" + str(save_name) + ".txt", "w") as f:
            f.write(output)

    return fb, g_score, detection_score