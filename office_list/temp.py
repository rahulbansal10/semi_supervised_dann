import fileinput

with fileinput.FileInput("webcam_list.txt", inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace("shuyang", "rahul/dataset"), end='')