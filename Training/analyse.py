'''
Author: Xiao Luo lxiao70@gatech.edu
Date: 2023-09-19 00:06:59
LastEditors: Xiao Luo lxiao70@gatech.edu
LastEditTime: 2023-09-22 22:56:20
FilePath: /TinyML2023EIC-Gatech/Training/train_result/analyse.py
'''
import csv
from collections import defaultdict
import os
import numpy as np
import re

# 指定CSV文件所在的目录
directory = './train_result/model_best'  # 你的CSV文件目录路径

# 创建一个默认字典，用于存储数据
data_dict = defaultdict(list)

# 获取目录中所有的CSV文件
def sort_key(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return 0
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
csv_files = sorted(csv_files, key=sort_key)

# 逐个处理CSV文件
for idx, csv_file in enumerate(csv_files):
    with open(os.path.join(directory, csv_file), 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # 将字符串转换为浮点数
            for key, value in row.items():
                if key != "model":
                    row[key] = float(value)
            # 使用(dense1, dense2)作为键，将整行数据添加到字典中
            key = row["model"]
            row["model_idx"] = idx
            data_dict[key].append(row)


avgScore = {}
for key, value in data_dict.items():
    train_scores = [v["trainScore"] for v in value]
    test_scores = [v["testScore"] for v in value]
    idx = np.argmax([train+test for train, test in zip(train_scores, test_scores)])
    analysis = {
        "avg_train": np.mean(train_scores),
        "std_train": np.std(train_scores),
        "avg_test": np.mean(test_scores),
        "std_test": np.std(test_scores),
        "max_combine": np.max([train+test for train, test in zip(train_scores, test_scores)]),
        "c_train": value[idx]["trainScore"],
        "c_test": value[idx]["testScore"],
        "max_idx": value[idx]["model_idx"],
    }
    avgScore[key] = analysis
print("Model Avg Train Std Train Avg Val Std Val Max Combine(Train+Test)       Max Idx")
f = open("./model_best.csv", "w")
f.write("Model,Avg Train,Std Train,Avg Val,Std Val,Max Combine(Train+Test),Max Idx\n")
for key, value in avgScore.items():
    model = key
    avg_train, std_train, avg_test, std_test = value["avg_train"], value["std_train"], value["avg_test"], value["std_test"]
    max_combine, ctrain, ctest = value["max_combine"], value["c_train"], value["c_test"]
    max_idx = value["max_idx"]
    print(f"{model:<10}{avg_train:<10.4f}{std_train:<10.4f}{avg_test:<8.4f}{std_test:<8.4f}{max_combine:<8.4f}({ctrain:.4f}+{ctest:.4f}){max_idx:<16}")
    f.write(f"{model},{avg_train},{std_train},{avg_test},{std_test},{max_combine}({ctrain}+{ctest}),{max_idx}\n")
f.close()