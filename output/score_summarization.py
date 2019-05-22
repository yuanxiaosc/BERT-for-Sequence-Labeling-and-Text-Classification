import os
# 查看当前工作目录
retval = os.getcwd()
print("当前工作目录为 %s" % retval)

path = "output/"

# 修改当前工作目录
os.chdir(path)

# 查看修改后的工作目录
retval = os.getcwd()
print("目录修改成功 %s" % retval)


file_names = os.listdir()
print(file_names)
score_summarization_f = open("model_score_summarization.txt", "w")

for file_name in file_names:
    if file_name[-4:] == "ckpt":
        log_file_path = os.path.join(file_name, "model_score_log.txt")
        if os.path.exists(log_file_path):
            score_summarization_f.write("*" * 100 + "\n")
            score_summarization_f.write("*" * 28 + file_name + "*" * 28 + "\n")
            score_summarization_f.write("*" * 100 + "\n")
            for line in open(log_file_path):
                score_summarization_f.write(line)

score_summarization_f.close()
