from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Input, Activation, Add, Dense
from keras.layers import Flatten, Conv1D, BatchNormalization

import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow_hub as hub
import numpy as np



def residual_block(y, nb_channels, _strides=(1), _project_shortcut=False):
    shortcut = y

    y = Conv1D(nb_channels, kernel_size=3, strides=_strides, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(nb_channels, kernel_size=3, strides=1, padding='same')(y)
    y = BatchNormalization()(y)

    # short cut reduce dimension
    if _project_shortcut or _strides != (1):
        shortcut = Conv1D(nb_channels, kernel_size=1, strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = Add()([shortcut, y])

    y = Activation('relu')(y)

    return y

def create_res_net(input_size):
    input = Input(shape=input_size)  # 这里的shape维度可能需要调整(int, 1)形式，int为时间信号长度

    # 85 32
    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # stage 1
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # stage 2
    x = residual_block(x, 128, _strides=2, _project_shortcut=True)
    x = residual_block(x, 128)

    # stage 3
    x = residual_block(x, 256, _strides=2, _project_shortcut=True)
    x = residual_block(x, 256)

    # stage 4
    x = residual_block(x, 512, _strides=2, _project_shortcut=True)
    x = residual_block(x, 512)

    # global evg pooling for finial layer
    x = Flatten()(x)
    
    x = Dense(2)(x)

    model = Model(input, x)

    return model


def create_resnet18_best(input_size):
    input = Input(shape=input_size)  # 这里的shape维度可能需要调整(int, 1)形式，int为时间信号长度

    # 85 32
    x = Conv1D(filters=3, kernel_size=85, strides=32, padding='valid', activation=None, use_bias=True)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # stage 1
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # stage 2
    x = residual_block(x, 128, _strides=2, _project_shortcut=True)
    x = residual_block(x, 128)

    # stage 3
    x = residual_block(x, 256, _strides=2, _project_shortcut=True)
    x = residual_block(x, 256)

    # stage 4
    x = residual_block(x, 512, _strides=2, _project_shortcut=True)
    x = residual_block(x, 512)

    # global evg pooling for finial layer
    x = Flatten()(x)
    
    x = Dense(2)(x)

    model = Model(input, x)

    return model


def draw(file_name, x):
    axis = list(range(0, len(x)))
    plt.clf()
    plt.plot(axis, x, marker='o', markersize=2)
    plt.savefig(file_name)


def delete_dir():
    directory_path = "./resnet/fig/"
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Error deleting files: {str(e)}")


def transform_to_imgs(x):
    tmp_dir = "./resnet/fig/"
    delete_dir()
    img_list = []
    for idx, sample in enumerate(x):
        file_name = f"{tmp_dir}/{idx}.jpg"
        draw(file_name, sample)
        img = load_images_from_directory(file_name)
        img_list.append(img)
    delete_dir()
    img_list = np.array(img_list)
    print(f"shape is {img_list.shape}")
    return img_list


def load_images_from_directory(file_name, input_size=(224, 224), batch_size=32, subject=None):
    def load_and_preprocess_image(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, input_size)
        return img
    
    # 使用map函数将图像加载和预处理应用于每个文件
    image = load_and_preprocess_image(file_name)
    
    return image


def resnet152():
    resnet152_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4"  # 预训练的 ResNet-152 模型
    model = tf.keras.Sequential([
        hub.KerasLayer(resnet152_url, input_shape=(224, 224, 3), trainable=True),  # 冻结预训练模型的权重
        tf.keras.layers.Dense(2)  # 输出层，适应你的二分类问题
    ])
    return model


