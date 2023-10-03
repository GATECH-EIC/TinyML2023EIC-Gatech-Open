import tensorflow as tf

# 加载TFLite模型
tflite_model_path = './train_ckpt/5featuresV3/45/16_6_model_features_2023-09-23-01:59:02.507473.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 获取模型的所有张量详情
tensor_details = interpreter.get_tensor_details()

# 创建字典来存储参数
relu_params = {}
dense_weights = {}
dense_biases = {}

# 用于跟踪上一层的类型
prev_layer_type = None

# 遍历所有张量详情
for detail in tensor_details:
    tensor_name = detail['name']
    print(detail)
    if "flatten" in tensor_name.lower():
        continue
    
    tensor_data = interpreter.get_tensor(detail['index'])

    # 解析张量名称以识别层类型
    if 'relu' in tensor_name.lower():
        # 这是ReLU层的参数
        relu_params[tensor_name] = tensor_data
    elif 'dense' in tensor_name.lower():
        # 这是Dense层的参数
        if prev_layer_type == 'batchnormalization':
            # 如果上一层是BatchNormalization，将其合并到Dense层的权重和偏置中
            dense_weights[tensor_name] = tensor_data * interpreter.get_tensor(prev_dense_weights_index)
            dense_biases[tensor_name] = (tensor_data - interpreter.get_tensor(prev_dense_weights_index)) * interpreter.get_tensor(prev_dense_biases_index)
        else:
            dense_weights[tensor_name] = tensor_data
    elif 'batchnormalization' in tensor_name.lower():
        # 这是BatchNormalization层的参数
        prev_layer_type = 'batchnormalization'
        prev_dense_weights_index = detail['index'] - 2  # 上一层的权重索引
        prev_dense_biases_index = detail['index'] - 1  # 上一层的偏置索引

# 打印参数
for relu_name, relu_data in relu_params.items():
    print(f'ReLU Parameter Name: {relu_name}')
    print(f'ReLU Parameter Data: {relu_data}')

for dense_name, dense_data in dense_weights.items():
    print(f'Dense Weight Name: {dense_name}')
    print(f'Dense Weight Data: {dense_data}')

for bias_name, bias_data in dense_biases.items():
    print(f'Dense Bias Name: {bias_name}')
    print(f'Dense Bias Data: {bias_data}')
