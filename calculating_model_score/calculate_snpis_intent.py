import os
import numpy as np
from sklearn_metrics_function import show_metrics

SNIPS_intent_label = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

with open(os.path.join("SNIPS_Intent", "label")) as label_f:
    label_list = [label.replace("\n", "") for label in label_f.readlines()]
    #print(len(label_list), label_list)

predit_label_value = np.fromfile(os.path.join("SNIPS_Intent", "test_results.tsv"), sep="\t")
predit_label_value = predit_label_value.reshape(-1, len(SNIPS_intent_label))
predit_label_value = np.argmax(predit_label_value, axis=1)
predit_label = [SNIPS_intent_label[label_index] for label_index in predit_label_value]

#print(len(predit_label), predit_label)


show_metrics(y_test=label_list, y_predict=predit_label, labels=SNIPS_intent_label)