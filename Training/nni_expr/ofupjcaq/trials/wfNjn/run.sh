#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='ofupjcaq'
export NNI_SYS_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ofupjcaq/trials/wfNjn'
export NNI_TRIAL_JOB_ID='wfNjn'
export NNI_OUTPUT_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ofupjcaq/trials/wfNjn'
export NNI_TRIAL_SEQ_ID='15'
export NNI_CODE_DIR='/data/hyou37/TinyML2023EIC-Gatech-Open/Training'
cd $NNI_CODE_DIR
eval 'python select_model_dev.py     --model model_features     --enable_nni True     --enable_fgs False     --parallel False' 1>/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ofupjcaq/trials/wfNjn/stdout 2>/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ofupjcaq/trials/wfNjn/stderr
echo $? `date +%s%3N` >'/data/hyou37/TinyML2023EIC-Gatech-Open/Training/nni_expr/ofupjcaq/trials/wfNjn/.nni/state'