#include "aiRun.h"

# define TRUE 1
# define FALSE 0

Matrix *Dense1_weight;
Matrix *Dense1_bias;
Matrix *Dense2_weight;
Matrix *Dense2_bias;
Matrix *Dense3_weight;
Matrix *Dense3_bias;

float dense1_weight_float[62][16];
float dense2_weight_float[16][6];
float dense3_weight_float[6][2];
float dense1_bias_float[16];
float dense2_bias_float[6];
float dense3_bias_float[2];

int compare (const void * a, const void * b)
{
    return ( *(float*)a - *(float*)b );
}

void cal_mean_std(const float* data, int len, float* mean_out, float* std_out){
/*
    @FUNCTION: calculate mean and std of a float array
    @PARAMETER:
        INPUT:
            data: input array
            len: length of input array
        OUTPUT:
            mean_out: mean of input array
            std_out: std of input array
*/
    *mean_out = 0.;
    *std_out = 0.;
    if (len == 0) {
        return;
    }

    for (int i=0; i<len; i++) {
        *mean_out += data[i];
    }
    *mean_out /= len;
    for (int i=0; i<len; i++) {
        float temp = (data[i] - *mean_out);
        *std_out += temp * temp;
    }
    *std_out  = sqrt((*std_out)/len);
}


void get_feature(const float* data, float limit, uint8_t up_or_down, float* feature_out){
/*
    @FUNCTION: get feature of input for one group of factor and threshold
    @PARAMETER: 
        INPUT:
            data: input array
            limit: limit for count a peak
            up_or_down: 0 for upPeak, 1 for downPeak
        OUTPUT:
            feature_out: feature of input array
                        length should be 5
                        {numPeak, peak_mean, peak_std, diff_mean, diff_std}
*/

    int peak_cnt = 0; // the number of peak
    int* peak_sampled = (int *) malloc(70 * sizeof(int));
    float* peak_value = (float *) malloc(70 * sizeof(float));

    // up Peak:
    if (up_or_down == 0){
        for (int i=0; i<INPUT_NUM; i++){
            if (data[i] > limit){
                peak_sampled[peak_cnt] = i;
                peak_value[peak_cnt] = data[i];
                peak_cnt += 1;
                // delay_steps = 0;
                i+=17;
            }
        }
    }
    // down Peak:
    else if (up_or_down == 1){
        for (int i=0; i<INPUT_NUM; i++){
            if (data[i] < -limit){
                peak_sampled[peak_cnt] = i;
                peak_value[peak_cnt] = data[i];
                peak_cnt += 1;
                // delay_steps = 0;
                i+=17;
            }
        }
    }

    // get peak diff array
    float peak_diff[peak_cnt-1];
    for (int i=0; i<peak_cnt-1; i++) {
        peak_diff[i] = (float)(peak_sampled[i+1] - peak_sampled[i]);
    }

    // calculate {peak_mean, peak_std, diff_mean, diff_std}
    cal_mean_std(peak_value, peak_cnt, feature_out+1, feature_out+2);
    cal_mean_std(peak_diff, peak_cnt-1, feature_out+3, feature_out+4);
    // feature_out[0] = peak_cnt;

    free(peak_sampled);
    free(peak_value);

    if (peak_cnt < 3) {
        feature_out[0] = peak_cnt;
        return;
    }

    // reject_outliers
    peak_cnt -= 1;
    // float* peak_diff_tmp = (float *) malloc((peak_cnt) * sizeof(float));
    float peak_diff_tmp[peak_cnt];
    float* d = peak_diff_tmp;
    memcpy(peak_diff_tmp, peak_diff, peak_cnt*4);
    qsort (peak_diff_tmp, peak_cnt, sizeof(float), compare);
    float peak_diff_median = (peak_cnt%2) ? peak_diff_tmp[peak_cnt/2] : (peak_diff_tmp[peak_cnt/2]+peak_diff_tmp[peak_cnt/2 - 1]) / 2;
    for (int i=0; i<peak_cnt; i++) {
        d[i] = fabsf(peak_diff[i] - peak_diff_median);
    }
		
		
    float d_tmp[peak_cnt];
    memcpy(d_tmp, d, peak_cnt*4);
    qsort (d_tmp, peak_cnt, sizeof(float), compare);
    float mdev = (peak_cnt%2) ? d_tmp[peak_cnt/2] : (d_tmp[peak_cnt/2]+d_tmp[peak_cnt/2 - 1]) / 2;
		
    float s;
    int trunc_peak_diff_count = 0;
    float trunc_peak_diff_sum = 0;
		
	for (int i=0; i<peak_cnt; i++) {
        s = mdev ? d[i]/mdev : 0.f;

        if (s < 2.f) {
            trunc_peak_diff_count += 1;
            trunc_peak_diff_sum += peak_diff[i];
        }
    }
    // Back to Count numPeaks (countPeaks)
    float robustAvgInterval = trunc_peak_diff_sum / trunc_peak_diff_count;
    float numPeaks = INPUT_NUM / robustAvgInterval;
    feature_out[0] = numPeaks;
    return;
}


void get_network_input(float* data, float* features){
/*
    @FUNCTION: get network input, it is features of input array
    @PARAMETER: 
        INPUT:
            data: input array, len: 1250
        OUTPUT:
            features: network input, features
                        length should be 67
*/

    float temp[5];
    cal_mean_std(data, INPUT_NUM, &features[61], &features[60]);
    for (int i=0; i<FACTOR_NUM; i++) {
        get_feature(data, factor[i] * features[60], 0, temp);
        features[i*12] = temp[0];
        features[i*12+1] = temp[0] > threshold[i] ? 1.0 : 0.0;
        memcpy(features+i*12+4, temp+1, 4*4);
			
        get_feature(data, factor[i] * features[60], 1, temp);
        features[i*12+2] = temp[0];
        features[i*12+3] = temp[0] > threshold[i] ? 1.0 : 0.0;
        memcpy(features+i*12+8, temp+1, 4*4);
    }
    return;
}



void dequant(union TwoFloat16 input, float *output1, float *output2){
    *output1 = input.parts.fraction1 / 16384.;
    *output2 = input.parts.fraction2 / 16384.;

    if(input.parts.exponent1) {
        *output1 += 1.0;
    }
    if(input.parts.exponent2) {
        *output2 += 1.0;
    }
    if(input.parts.sign_bit1) {
        *output1 = -(*output1);
    }
    if(input.parts.sign_bit2) {
        *output2 = -(*output2);
    }
}

void deconvert(union TwoFloat16 *src, float* dest, int row) {
    row /= 2;
    for(int i=0;i<row;++i) {
        dequant(*(src + i), dest+i*2, dest+i*2+1);
    }
}



void Model_Init(){
    deconvert(dense1_wq,dense1_weight_float,16 * 62);
    deconvert(dense2_wq,dense2_weight_float,6 * 16);
    deconvert(dense3_wq,dense3_weight_float,2 * 6);
    deconvert(dense1_bq,dense1_bias_float,16);
    deconvert(dense2_bq,dense2_bias_float,6);
    deconvert(dense3_bq,dense3_bias_float,2);

    for(int i=0;i<62;++i) {
        for(int j=0;j<16;++j) {
            dense1_weight_float[i][j] /= 3.75;
        }
    }

    Dense1_weight = Matrix_gen(16, 62, (float*) dense1_weight_float);
    Dense1_bias = Matrix_gen(16, 1, (float*) dense1_bias_float);

    Dense2_weight = Matrix_gen(6, 16, (float*) dense2_weight_float);
    Dense2_bias = Matrix_gen(6,1, (float*) dense2_bias_float);

    Dense3_weight = Matrix_gen(2, 6, (float*) dense3_weight_float);
    Dense3_bias = Matrix_gen(2,1, (float*) dense3_bias_float);
}

int aiRun(const float *input, float *result){

    // get features for network input
    float features[FACTOR_NUM*12+2];
    get_network_input(input, features);
    // Layer 1
    Matrix *Result;
    Matrix *Result_old;
    Matrix *Features = Matrix_gen(62, 1, (float*) features); 
    Result = M_mul(Dense1_weight, Features);
    Result_old = Result;
    free(Features);

    Result = M_add(Result, Dense1_bias);
    Result = M_relu(Result);

    // Layer 2
    Result = M_mul(Dense2_weight, Result);
    free(Result_old->data);
    free(Result_old);
    Result_old = Result;

    Result = M_add(Result, Dense2_bias);
    Result = M_relu(Result);

    // Layer 3
    Result = M_mul(Dense3_weight, Result);
    free(Result_old->data);
    free(Result_old);
    Result_old = Result;

    Result = M_add(Result, Dense3_bias);


    // output
    result[0] = Result->data[0];
    result[1] = Result->data[1];

    free(Result->data);
    free(Result);

    return 0;
}
