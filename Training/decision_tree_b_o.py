import numpy as np
import tensorflow as tf

FACTORS =    [1.5, 1.75, 2.0, 2.25, 2.5]
THRESHOLDS = [9.191, 9.21, 9.263, 9.157, 9.154]

def dt_infer(x, data_paths, factors=FACTORS, thresholds=THRESHOLDS, ensemble=True):
    y = []

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
        print(f"\r running {iter+1}/{len(x)}", end="")
    return np.array(y)


def countPeaks(chunk,factor, inp_len=1250):
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


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    ans = data[s<m]
    if ans.ndim != 1:
        ans = np.squeeze(ans)
    return ans
