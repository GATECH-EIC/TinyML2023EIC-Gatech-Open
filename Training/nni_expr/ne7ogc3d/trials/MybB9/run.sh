#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='ne7ogc3d'
export NNI_SYS_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ne7ogc3d/trials/MybB9'
export NNI_TRIAL_JOB_ID='MybB9'
export NNI_OUTPUT_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ne7ogc3d/trials/MybB9'
export NNI_TRIAL_SEQ_ID='6'
export NNI_CODE_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training'
cd $NNI_CODE_DIR
eval 'python select_model_dev.py     --model model_features     --enable_nni True' 1>/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ne7ogc3d/trials/MybB9/stdout 2>/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ne7ogc3d/trials/MybB9/stderr
echo $? `date +%s%3N` >'/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ne7ogc3d/trials/MybB9/.nni/state'