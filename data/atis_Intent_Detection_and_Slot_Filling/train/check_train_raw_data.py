label_data = open("label", encoding='utf-8').readlines()
label_data = [x.strip() for x in label_data]
print(len(label_data))
label_kinds = set(label_data)
print(label_kinds)