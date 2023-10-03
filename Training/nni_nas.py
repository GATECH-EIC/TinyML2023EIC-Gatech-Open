from nni.experiment import Experiment


search_space_model_features = {
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.3]},
    'epoch': {'_type': 'quniform', '_value': [35, 70, 1]},
    "dropout1": {'_type': 'uniform', '_value': [0.1, 0.3]},
    "dropout2": {'_type': 'uniform', '_value': [0.1, 0.3]},
    "dense1": {'_type': 'quniform', '_value': [2, 10, 1]},
    "dense2": {'_type': 'quniform', '_value': [2, 10, 1]},
    'factor_0': {'_type': 'uniform', '_value': [1, 4]},
    'factor_1': {'_type': 'uniform', '_value': [1, 4]},
    "factor_2": {'_type': 'uniform', '_value': [1, 4]},
    "factor_3": {'_type': 'uniform', '_value': [1, 4]},
    "factor_4": {'_type': 'uniform', '_value': [1, 4]},
    'threshold_0': {'_type': 'uniform', '_value': [7, 13]},
    'threshold_1': {'_type': 'uniform', '_value': [7, 13]},
    "threshold_2": {'_type': 'uniform', '_value': [7, 13]},
    "threshold_3": {'_type': 'uniform', '_value': [7, 13]},
    "threshold_4": {'_type': 'uniform', '_value': [7, 13]},
}

command_modelfeatures = "python select_model_dev.py \
    --model model_features \
    --enable_nni True \
    --enable_fgs False \
    --parallel False"

experiment = Experiment('local')
experiment.config.experiment_name = 'nni_tinyml_test'
experiment.config.trial_command = command_modelfeatures
experiment.config.trial_code_directory = '.'
experiment.config.experiment_working_directory = './nni_expr'
experiment.config.search_space = search_space_model_features
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {
        'optimize_mode': 'maximize',
}
experiment.config.max_trial_number = 1500  # 最多尝试实验个数
experiment.config.trial_concurrency = 10  # 同时实验个数

experiment.run(port=23400)
input() # avoid experiment stop when tasks done
experiment.stop()
