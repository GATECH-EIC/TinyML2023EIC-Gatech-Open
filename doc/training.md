# Training

## Set up environment
```bash
cd Training
conda env create -f environment.yml
conda activate tinyml
```

## Find hyper-parameters
In this section, we need to find the network structure and hyper-parameters for feature extraction module.  
We use [nni](https://nni.readthedocs.io/en/stable/) with Tree-structured Parzen Estimator tunner to search for our network architecture and feature extraction hyper-parameters.
Run  
```bash
python nni_nas.py
```
Then you can open the link output in the terminal to check best hyper-parameter.  
We also provide one [selected hyper-parameters](../Training/nni_params/features_extr_5v3.json).  

## Train more models with selected hyper-parameters and select the best one
Run  
```bash
python train.py --param_path <hyper-parameter, store in json file>
# For example, you can run
# python train.py --param_path ./nni_params/features_extr_5v3.json
```
This will train 1000 models with selected hyper-parameters, create dirs and store results in:  
> + detection performance: train_result/model_best/{model_index}
> + model weights in tflite: train_ckpt/model_best/{model_index}
> + training logs: log/model_best/{model_index}

In train_ckpt/model_best, we have:  
> + FLOAT16.tflite model with float16 quantization in tflite format
> + INT8.tflite model with quant aware training in int8 in tflite format
> + NONE.tflite model without quantization in tflite format
> + Original.h5 model without quantization in keras format

We also provide a simple analyse script, run  
```bash
python analyse.py
```
This will output detection performance report and stored in train_result/model_best/model_best.csv.   

