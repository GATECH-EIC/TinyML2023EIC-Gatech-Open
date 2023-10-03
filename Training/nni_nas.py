from nni.experiment import Experiment


# This is for nni_model
search_space_cluster = {
    'epoch': {'_type': 'quniform', '_value': [1, 30, 1]},
    "use_swa": {"_type": "choice", "_value": [0, 1]},
    "centroid": {"_type": "choice", "_value": ["Linear", "Density", "KPP"]},
    "num_cluster": {'_type': 'quniform', '_value': [8, 256, 1]}
}


search_space_model_features = {
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.3]},
    'epoch': {'_type': 'quniform', '_value': [35, 70, 1]},
    "dropout1": {'_type': 'uniform', '_value': [0.1, 0.3]},
    "dropout2": {'_type': 'uniform', '_value': [0.1, 0.3]},
    "dropout3": {'_type': 'uniform', '_value': [0.1, 0.3]},
    "dense1": {'_type': 'quniform', '_value': [2, 10, 1]},
    "dense2": {'_type': 'quniform', '_value': [2, 10, 1]},
    "dense3": {'_type': 'quniform', '_value': [0, 10, 1]},
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
    # "detect_gap": {'_type': 'quniform', '_value': [30, 40, 1]},
    # "n_features": {'_type': 'quniform', '_value': [15, 30, 1]},
    # "enable_normalization": {'_type': 'quniform', '_value': [0, 1, 1]}
}


search_space_nnimodel0 = {
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'num_conv': {'_type': 'quniform', '_value': [3, 4, 1]},
    "kernel_0": {'_type': 'quniform', '_value': [30, 80, 1]},
    "strides_0": {'_type': 'quniform', '_value': [16, 32, 1]},
    "kernel_1": {'_type': 'quniform', '_value': [50, 100, 1]},
    "strides_1": {'_type': 'quniform', '_value': [32, 64, 1]},
    "kernel_2": {'_type': 'quniform', '_value': [75, 150, 1]},
    "strides_2": {'_type': 'quniform', '_value': [32, 100, 1]},
    "kernel_3": {'_type': 'quniform', '_value': [30, 150, 1]},
    "strides_3": {'_type': 'quniform', '_value': [16, 100, 1]},
    "dropout1": {'_type': 'uniform', '_value': [0.1, 0.3]},
    "dropout2": {'_type': 'uniform', '_value': [0.1, 0.3]},
    "dense1": {'_type': 'quniform', '_value': [10, 30, 1]},
    "dense2": {'_type': 'quniform', '_value': [5, 15, 1]},
}

parameter_size = 15
search_space_dt = {
    "parameter_size": {'_type': 'quniform', '_value': [parameter_size, parameter_size, 1]}
}

for i in range(parameter_size):
    search_space_dt[f"factor_{i}"] = {'_type': 'uniform', '_value': [1, 4]}
    search_space_dt[f"threshold_{i}"] = {'_type': 'uniform', '_value': [7, 13]}

command_modelfeatures = "python select_model_dev.py \
    --model model_features \
    --enable_nni True \
    --enable_fgs False \
    --parallel False"

command_dt = "python select_model.py \
    --decision_tree True \
    --enable_nni True"

command_cluster = "python train_cluster.py \
    --param_path ./nni_params/features_extr_5v3.json \
    --enable_nni True"


experiment = Experiment('local')
experiment.config.experiment_name = 'nni_tinyml_test'
experiment.config.trial_command = command_modelfeatures
experiment.config.trial_code_directory = '.'
experiment.config.experiment_working_directory = './nni_expr'
experiment.config.search_space = search_space_model_features
# experiment.config.tuner.name = 'Evolution'
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {
        'optimize_mode': 'maximize',
        # 'population_size': 150
}
experiment.config.max_trial_number = 1500  # 最多尝试实验个数
experiment.config.trial_concurrency = 10  # 同时实验个数
# experiment.id = "codbwux6"

experiment.run(port=23400)

input()

experiment.stop()
