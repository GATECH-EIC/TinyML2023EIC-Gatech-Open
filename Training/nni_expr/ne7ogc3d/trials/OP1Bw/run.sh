#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='ne7ogc3d'
export NNI_SYS_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ne7ogc3d/trials/OP1Bw'
export NNI_TRIAL_JOB_ID='OP1Bw'
export NNI_OUTPUT_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ne7ogc3d/trials/OP1Bw'
export NNI_TRIAL_SEQ_ID='3'
export NNI_CODE_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training'
cd $NNI_CODE_DIR
eval 'python select_model_dev.py     --model model_features     --enable_nni True' 1>/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ne7ogc3d/trials/OP1Bw/stdout 2>/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ne7ogc3d/trials/OP1Bw/stderr
echo $? `date +%s%3N` >'/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ne7ogc3d/trials/OP1Bw/.nni/state'