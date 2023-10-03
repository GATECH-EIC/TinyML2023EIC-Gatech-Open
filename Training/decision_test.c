/*import numpy as np
std = np.array([])
chunk = []
factor = 2
idx = 0
length = 1250
peak_threshold = std.item() * factor
trough_sampled_idx = []
peak_sampled_idx = []
detect_gap = 20
while idx < length:
    val = chunk[idx]
    if val > peak_threshold:
        peak_sampled_idx.append(idx)
        idx += detect_gap
*/

#include <stdlib.h>
int* countPeaks(float std, float factor, int*chunk){
    const int BIAS = sizeof(int);
    const int LENGTH = 1250;
    int *peak_sampled_idx = malloc(BIAS*65);
    int idx = 0;
    const int detect_gap = 20;
    int peak_threshold = std * factor;
    int peak_cnt = 0;
    while(idx < LENGTH) {
        if (chunk[idx] > peak_threshold) {
            peak_sampled_idx[peak_cnt] = idx;
            ++peak_cnt;
            idx += detect_gap;
        } else {
            idx += 1;
        }
    }
}