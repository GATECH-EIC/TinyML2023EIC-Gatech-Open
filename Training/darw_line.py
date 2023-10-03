import matplotlib.pyplot as plt

# 指定要读取的文件名
sample_file = "S02-VFb-150" + ".txt"
file_name = "./tinyml_contest_data_training/" + sample_file  # 替换为您的文件名

numbers = []

# 打开文件并读取数字
try:
    with open(file_name, "r") as file:
        for i in range(1250):
            line = file.readline()
            # 将每一行的数字转换为整数，并添加到列表中
            number = float(line.strip())
            numbers.append(number)

    # 绘制图形
    plt.plot(numbers)
    plt.xlabel("Index")
    plt.ylabel("Number")
    plt.title("Plot of Numbers")
    plt.grid(True)
    plt.savefig("./plots/wrong/" + sample_file + ".jpg")

except FileNotFoundError:
    print(f"文件 '{file_name}' 未找到。请确保文件名正确。")
except ValueError:
    print("文件中包含无效的数字。请确保文件中的每一行都包含有效的整数。")
except Exception as e:
    print(f"发生错误：{e}")
