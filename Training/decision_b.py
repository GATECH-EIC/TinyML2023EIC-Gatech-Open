import numpy as np
import tensorflow as tf

FACTORS =    [1.5, 1.75, 2.0, 2.25, 2.5]
THRESHOLDS = [9.191, 9.21, 9.263, 9.157, 9.154]
# THRESHOLDS = [12.191, 12.21, 12.263, 12.157, 12.154]

def dt_infer(x, data_paths, nni_params, factors=FACTORS, thresholds=THRESHOLDS, ensemble=True):
    y = []

    parameter_size = int(nni_params["parameter_size"])
    factors = [float(nni_params[f"factor_{i}"]) for i in range(parameter_size)]
    thresholds = [float(nni_params[f"threshold_{i}"]) for i in range(parameter_size)]

    for iter, sample in enumerate(x):
        results = []
        va_pred_cnt = []
        data_path = data_paths[iter].numpy().decode('utf-8')
        
        if ensemble:
            for i in range(len(thresholds)):
                thresh=thresholds[i]
                factor_=factors[i]
                sample = sample.squeeze()
                numPeaks = countPeaks(sample,factor_)
                # numUnpeaks, repeated_unpeaks = count_unpeak(sample, factor_)
                results.append(numPeaks)
                va_pred_cnt.append(numPeaks>thresh)
            # majority voting on ensemble results
            # qrs = calculate_qrs_duration(sample)
            if np.count_nonzero(va_pred_cnt) > (len(factors)/2+1):
                pred = 1
            else:
                pred = 0
        else:
            thresh=thresholds[-1]
            factor_=factors[-1]
            sample = sample.squeeze()
            numPeaks = countPeaks(sample,factor_)
            if numPeaks > thresh:
                pred = 1
            else:
                pred = 0
        y.append(pred)
        if nni_params is None:
            print(f"\r running {iter+1}/{len(x)}", end="")
    return np.array(y)


def countPeaks(chunk,factor,idxx,inp_len=1250):
    std = np.std(chunk)
    mean = np.mean(chunk)

    # detect flag prevents over-sampling near the peaks
    detect = True
    delay_steps = 0
    peak_cnt = 0
    peak_sampled_idx = []
    # iterate through the input and count peaks
    for idx, val in enumerate(chunk):
        peak_threshold = std.item() * factor
        if val > peak_threshold:
            if detect:
                peak_sampled_idx.append(idx)
                peak_cnt += 1
                delay_steps = 0
                detect = False
        if not detect:
            delay_steps += 1
            if delay_steps > 20:
                detect = True
    # truncation works only when three or more peaks are sampled
    if len(peak_sampled_idx) < 3:
        return len(peak_sampled_idx)
    # baseline method
    og_ans = len(peak_sampled_idx)
    # truncation method
    peak_diff = np.diff(np.asarray(peak_sampled_idx))
    trunc_peak_diff = reject_outliers(peak_diff)
    trunc_ans = trunc_peak_diff.size + 1
    # interval method
    robustAvgInterval = sum(trunc_peak_diff)/trunc_peak_diff.size
    inter_ans = inp_len/robustAvgInterval

    return inter_ans

'''
def count_unpeak(chunk,factor,inp_len=1250):
    std = np.std(chunk)
    mean = np.mean(chunk)

    # detect flag prevents over-sampling near the peaks
    detect = True
    delay_steps = 0
    peak_cnt = 0
    peak_sampled_idx = []
    repeated_peaks = []
    # iterate through the input and count peaks
    for idx, val in enumerate(chunk):
        peak_threshold = std.item() * factor
        if val < -peak_threshold:
            if detect:
                peak_sampled_idx.append(idx)
                peak_cnt += 1
                delay_steps = 0
                detect = False
                repeated_peaks.append(0)
            else:
                repeated_peaks[-1] += 1
        if not detect:
            delay_steps += 1
            if delay_steps > 20:
                detect = True
    # truncation works only when three or more peaks are sampled
    if len(peak_sampled_idx) < 3:
        return len(peak_sampled_idx), []
    # baseline method
    og_ans = len(peak_sampled_idx)
    # truncation method
    peak_diff = np.diff(np.asarray(peak_sampled_idx))
    trunc_peak_diff = reject_outliers(peak_diff)
    trunc_ans = trunc_peak_diff.size + 1
    # interval method
    robustAvgInterval = sum(trunc_peak_diff)/trunc_peak_diff.size
    inter_ans = inp_len/robustAvgInterval

    if len(repeated_peaks) >= 2:
        repeated_peaks = repeated_peaks[1:-1]

    return inter_ans, repeated_peaks
'''

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    ans = data[s<m]
    if ans.ndim != 1:
        ans = np.squeeze(ans)
    return ans


import numpy as np
from scipy.signal import find_peaks
import scipy

def pan_tompkins(iegm, sampling_rate=250):
    
    # 带通滤波器参数
    lowcut = 0.5
    highcut = 50.0
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    # 带通滤波
    b, a = scipy.signal.butter(1, [low, high], btype='band')
    filtered = scipy.signal.lfilter(b, a, iegm)

    # 微分
    derivative = np.diff(filtered)

    # 平方
    squared = derivative ** 2

    # 积分（在这个例子中是通过移动平均实现的）
    window_width = int(0.150 * sampling_rate)
    integrated = np.convolve(squared, np.ones(window_width))

    # 阈值设定和QRS检测，这个例子中使用简单的绝对阈值
    threshold = np.mean(integrated) + 2 * np.std(integrated)

    qrs_complexes = (integrated > threshold).astype(int)

    return qrs_complexes

def calculate_qrs_duration(iegm, sampling_rate=250):
    qrs_lengths = []
    start = None
    qrs_complexes = pan_tompkins(iegm)
    for i in range(1, len(qrs_complexes)):
        if qrs_complexes[i-1] == 0 and qrs_complexes[i] == 1:
            start = i
        elif qrs_complexes[i-1] == 1 and qrs_complexes[i] == 0:
            if start is not None:
                qrs_lengths.append((i-start)/sampling_rate)
                start = None

    return qrs_lengths


'''
with time-comsuming but useless SVT_length
def countPeaksNN(chunk,factor, SVT_length, unpeak_lowerbound, unpeak_upperbound
               ,std, mean, inp_len=1250):

    # detect flag prevents over-sampling near the peaks
    detect = True
    delay_steps = 0
    peak_cnt = 0
    peak_sampled_idx = []
    repeated_peaks = []
    first_low = -1
    second_low = -1
    high = -1
    valid_SVT = 0
    # iterate through the input and count peaks
    for idx, val in enumerate(chunk):
        peak_threshold = std.item() * factor
        if val < -peak_threshold:
            if first_low == -1:
                first_low = idx
            elif idx - first_low <= 5:
                if val < chunk[first_low]:
                    first_low = idx
            elif second_low == -1 and idx - first_low >= 5:
                second_low = idx
                if second_low - first_low <= SVT_length and first_low < high < second_low:
                    while True:
                        if second_low + 1 >= len(chunk) or chunk[second_low+1] > chunk[second_low]:
                            break
                        second_low += 1
                    
                    while True:
                        if first_low + 1 >= len(chunk) or chunk[first_low+1] > chunk[first_low]:
                            break
                        first_low += 1

                    # try to diff VT and SVT
                    if unpeak_lowerbound <= abs(chunk[second_low]/chunk[first_low]) <= unpeak_upperbound:
                        valid_SVT += 1
                    first_low = second_low = high = -1
            else:
                first_low = second_low = -1
        if val > peak_threshold:
            if detect:
                peak_sampled_idx.append(idx)
                peak_cnt += 1
                delay_steps = 0
                detect = False
                repeated_peaks.append(0)
                high = idx
            else:
                repeated_peaks[-1] += 1
        if not detect:
            delay_steps += 1
            if delay_steps > 20:
                detect = True
                high = first_low = second_low = -1
        last_peaks = val
    # truncation works only when three or more peaks are sampled
    if len(peak_sampled_idx) < 3:
        return len(peak_sampled_idx), peak_sampled_idx, valid_SVT
    # baseline method
    og_ans = len(peak_sampled_idx)
    # truncation method
    peak_diff = np.diff(np.asarray(peak_sampled_idx))
    trunc_peak_diff = reject_outliers(peak_diff)
    trunc_ans = trunc_peak_diff.size + 1
    # interval method
    robustAvgInterval = sum(trunc_peak_diff)/trunc_peak_diff.size
    inter_ans = inp_len/robustAvgInterval

    if len(repeated_peaks) >= 2:
        repeated_peaks = repeated_peaks[1:-1]

    return inter_ans, peak_sampled_idx, valid_SVT
'''


def countPeaksNN(chunk,factor, std, detect_gap=20, inp_len=1250):
    peak_sampled_idx = []
    # iterate through the input and count peaks
    peak_threshold = std.item() * factor
    idx = 0
    while idx < inp_len:
        if chunk[idx] > peak_threshold:
            peak_sampled_idx.append(idx)
            idx += detect_gap
        else:
            idx += 1
    # truncation works only when three or more peaks are sampled
    if len(peak_sampled_idx) < 3:
        return len(peak_sampled_idx), peak_sampled_idx
    # baseline method
    og_ans = len(peak_sampled_idx)
    # truncation method
    peak_diff = np.diff(np.asarray(peak_sampled_idx))
    trunc_peak_diff = reject_outliers(peak_diff)
    trunc_ans = trunc_peak_diff.size + 1
    # interval method
    robustAvgInterval = sum(trunc_peak_diff)/trunc_peak_diff.size
    inter_ans = inp_len/robustAvgInterval

    # print(cur_detect_gap)


    return inter_ans, peak_sampled_idx


def count_unpeak(chunk,factor,std, detect_gap=20,inp_len=1250):
    peak_cnt = 0
    peak_sampled_idx = []
    repeated_peaks = []
    # iterate through the input and count peaks
    idx = 0
    peak_threshold = std.item() * factor
    while idx < inp_len:
        if chunk[idx] < -peak_threshold:
            peak_sampled_idx.append(idx)
            idx += detect_gap
        else:
            idx += 1
    # truncation works only when three or more peaks are sampled
    if len(peak_sampled_idx) < 3:
        return len(peak_sampled_idx), peak_sampled_idx
    # baseline method
    og_ans = len(peak_sampled_idx)
    # truncation method
    peak_diff = np.diff(np.asarray(peak_sampled_idx))
    trunc_peak_diff = reject_outliers(peak_diff)
    trunc_ans = trunc_peak_diff.size + 1
    # interval method
    robustAvgInterval = sum(trunc_peak_diff)/trunc_peak_diff.size
    inter_ans = inp_len/robustAvgInterval

    if len(repeated_peaks) >= 2:
        repeated_peaks = repeated_peaks[1:-1]

    return inter_ans, peak_sampled_idx