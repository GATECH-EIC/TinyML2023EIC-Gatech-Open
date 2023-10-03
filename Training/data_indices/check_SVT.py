

with open("train_indice.csv", 'r') as file:
    file.readline()
    positive = set()
    contain_SVT = set()
    while True:
        line = file.readline()
        if line == "":
            break
        label, part = line.split(",")
        subject, category, _ = part.split("-")
        if category == "SVT":
            contain_SVT.add(subject)
        elif label == "1":
            positive.add(subject)
    print(f"result is {contain_SVT & positive}")

        