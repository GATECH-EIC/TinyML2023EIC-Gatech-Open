import os

# 两个目录的路径
directory1 = './tinyml_contest_data_training_2022'
directory2 = './tinyml_contest_data_training_2023'

# 获取两个目录中的文件列表
files1 = os.listdir(directory1)
files2 = os.listdir(directory2)

# 获取两个目录中的共同文件名
common_files = set(files1) & set(files2)

# 初始化一个变量来统计不同文件的数量
different_files_count = 0

# 遍历共同文件并比较它们的内容
for idx, file_name in enumerate(common_files):
    if idx % 100 == 0:
        print(f"\rprocessing {idx}/{len(common_files)}", end="")
    file1_path = os.path.join(directory1, file_name)
    file2_path = os.path.join(directory2, file_name)

    with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2:
        content1 = file1.read()
        content2 = file2.read()

    content1 = content1.decode()
    content2 = content2.decode()

    content1 = str(content1).split("\n")
    content1.pop()
    content1 = [float(c.strip()) for c in content1]
    content2 = str(content2).split("\n")
    content2.pop()
    content2 = [float(c.strip()) for c in content2]
    if content1 != content2:
        for i in range(len(content1)):
            print(f"{content1[i]} {content2[i]}")
        print(f"文件 '{file_name}' 的内容不同.")
        print(len(content1))
        different_files_count += 1

print("\r")
if different_files_count == 0:
    print("两个目录中的所有同名文件的内容都相同.")
else:
    print(f"共有 {different_files_count} 个不同的文件.")
