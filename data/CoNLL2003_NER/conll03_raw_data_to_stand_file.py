import os

def get_examples(input_file):
    """Reads a BIO data."""
    with open(input_file) as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue
            if len(contends) == 0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label)
        seq_in = [line[1] for line in lines]
        seq_out = [line[0] for line in lines]
        return seq_in, seq_out

def conll03_raw_data_to_stand(path_to_raw_file=None):
    os.makedirs("train")
    os.makedirs("valid")
    os.makedirs("test")
    for file_type in ["train", "dev", "test"]:
        raw_file = file_type+".txt"
        seq_in, seq_out = get_examples(raw_file)
        if file_type=="dev":
            file_type="valid"
        with open(os.path.join(file_type, "seq.in"), "w") as seq_in_f:
            with open(os.path.join(file_type, "seq.out"), "w") as seq_out_f:
                for seq in seq_in:
                    seq_in_f.write(seq + "\n")
                for seq in seq_out:
                    seq_out_f.write(seq+"\n")



conll03_raw_data_to_stand()
