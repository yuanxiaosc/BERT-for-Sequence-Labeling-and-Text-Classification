import os
import numpy as np
from sklearn_metrics_function import show_metrics

ATIS_intent_label = ['atis_abbreviation', 'atis_aircraft', 'atis_aircraft#atis_flight#atis_flight_no',
                'atis_airfare', 'atis_airfare#atis_flight', 'atis_airfare#atis_flight_time',
                'atis_airline', 'atis_airline#atis_flight_no', 'atis_airport', 'atis_capacity',
                'atis_cheapest', 'atis_city', 'atis_day_name', 'atis_distance', 'atis_flight',
                'atis_flight#atis_airfare', 'atis_flight#atis_airline', 'atis_flight_no',
                'atis_flight_no#atis_airline', 'atis_flight_time', 'atis_ground_fare',
                'atis_ground_service', 'atis_ground_service#atis_ground_fare', 'atis_meal',
                'atis_quantity', 'atis_restriction']

with open(os.path.join("ATIS_Intent", "label")) as label_f:
    label_list = [label.replace("\n", "") for label in label_f.readlines()]
    #print(len(label_list), label_list)

predit_label_value = np.fromfile(os.path.join("ATIS_Intent", "test_results.tsv"), sep="\t")
predit_label_value = predit_label_value.reshape(-1, len(ATIS_intent_label))
predit_label_value = np.argmax(predit_label_value, axis=1)
predit_label = [ATIS_intent_label[label_index] for label_index in predit_label_value]

#print(len(predit_label), predit_label)


show_metrics(y_test=label_list, y_predict=predit_label, labels=ATIS_intent_label)