import os

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
            for line in  open(log_file_path):
                score_summarization_f.write(line)

score_summarization_f.close()
