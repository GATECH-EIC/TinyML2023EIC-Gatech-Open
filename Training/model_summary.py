import tensorflow as tf
from tensorflow import keras

def summarize_tflite_model(input_size, dense1, dense2):
    model = keras.Sequential([
            keras.layers.Input(shape=input_size),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(dense1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(dense2),
            keras.layers.ReLU(),
            keras.layers.Dense(2),
        ])

    # 打印模型摘要信息
    model.summary()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("用法: python summarize_tflite_model.py <input_size> <dense1> <dense2>")
        sys.exit(1)

    summarize_tflite_model(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
