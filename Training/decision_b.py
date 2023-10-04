import numpy as np

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    ans = data[s<m]
    if ans.ndim != 1:
        ans = np.squeeze(ans)
    return ans


def countPeaksNN(chunk,factor, std, detect_gap=20, inp_len=1250):
    peak_sampled_idx = []
    peak_threshold = std.item() * factor
    idx = 0
    while idx < inp_len:
        if chunk[idx] > peak_threshold:
            peak_sampled_idx.append(idx)
            idx += detect_gap
        else:
            idx += 1
    if len(peak_sampled_idx) < 3:
        return len(peak_sampled_idx), peak_sampled_idx
    peak_diff = np.diff(np.asarray(peak_sampled_idx))
    trunc_peak_diff = reject_outliers(peak_diff)
    robustAvgInterval = sum(trunc_peak_diff)/trunc_peak_diff.size
    inter_ans = inp_len/robustAvgInterval
    return inter_ans, peak_sampled_idx


def count_unpeak(chunk,factor,std, detect_gap=20,inp_len=1250):
    peak_sampled_idx = []
    repeated_peaks = []
    idx = 0
    peak_threshold = std.item() * factor
    while idx < inp_len:
        if chunk[idx] < -peak_threshold:
            peak_sampled_idx.append(idx)
            idx += detect_gap
        else:
            idx += 1
    if len(peak_sampled_idx) < 3:
        return len(peak_sampled_idx), peak_sampled_idx
    peak_diff = np.diff(np.asarray(peak_sampled_idx))
    trunc_peak_diff = reject_outliers(peak_diff)
    robustAvgInterval = sum(trunc_peak_diff)/trunc_peak_diff.size
    inter_ans = inp_len/robustAvgInterval

    if len(repeated_peaks) >= 2:
        repeated_peaks = repeated_peaks[1:-1]

    return inter_ans, peak_sampled_idx