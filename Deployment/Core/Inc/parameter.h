#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <stdint.h>

#define FACTOR_NUM 5
#define THRESHOLD_NUM 5
#define INPUT_NUM 1250

extern const float factor[5];
extern const float threshold[5];
// extern float input[1250];

// extern const float dense1_weight[62][16];
// extern const float dense1_bias[16];

// extern const float dense2_weight[16][6];
// extern const float dense2_bias[6];

// extern const float dense3_weight[6][2];
// extern const float dense3_bias[2];

union TwoFloat16 {
    uint32_t data;
    struct {
        uint32_t sign_bit1 : 1;     
        uint32_t exponent1 : 1;    
        uint32_t fraction1 : 14;    
        uint32_t sign_bit2 : 1;    
        uint32_t exponent2 : 1;     
        uint32_t fraction2 : 14;   
    } parts;
};
extern const union TwoFloat16 dense1_wq[16][31];
extern const union TwoFloat16 dense1_bq[8];
extern const union TwoFloat16 dense2_wq[6][8];
extern const union TwoFloat16 dense2_bq[3];
extern const union TwoFloat16 dense3_wq[2][3];
extern const union TwoFloat16 dense3_bq[1];

#endif
