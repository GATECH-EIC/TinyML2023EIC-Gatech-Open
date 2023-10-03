#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='gw120yfn'
export NNI_SYS_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/gw120yfn/trials/Bm4s9'
export NNI_TRIAL_JOB_ID='Bm4s9'
export NNI_OUTPUT_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/gw120yfn/trials/Bm4s9'
export NNI_TRIAL_SEQ_ID='16'
export NNI_CODE_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training'
cd $NNI_CODE_DIR
eval 'python select_model_dev.py     --model model_features     --enable_nni True' 1>/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/gw120yfn/trials/Bm4s9/stdout 2>/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/gw120yfn/trials/Bm4s9/stderr
echo $? `date +%s%3N` >'/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/gw120yfn/trials/Bm4s9/.nni/state'