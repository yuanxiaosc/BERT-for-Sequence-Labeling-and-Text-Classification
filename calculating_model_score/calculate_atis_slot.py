import os
from sklearn_metrics_function import show_metrics,delete_both_sides_is_O_word

ATIS_slot_label = ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', 'B-aircraft_code', 'B-airline_code', 'B-airline_name', 'B-airport_code', 'B-airport_name', 'B-arrive_date.date_relative', 'B-arrive_date.day_name', 'B-arrive_date.day_number', 'B-arrive_date.month_name', 'B-arrive_date.today_relative', 'B-arrive_time.end_time', 'B-arrive_time.period_mod', 'B-arrive_time.period_of_day', 'B-arrive_time.start_time', 'B-arrive_time.time', 'B-arrive_time.time_relative', 'B-booking_class', 'B-city_name', 'B-class_type', 'B-compartment', 'B-connect', 'B-cost_relative', 'B-day_name', 'B-day_number', 'B-days_code', 'B-depart_date.date_relative', 'B-depart_date.day_name', 'B-depart_date.day_number', 'B-depart_date.month_name', 'B-depart_date.today_relative', 'B-depart_date.year', 'B-depart_time.end_time', 'B-depart_time.period_mod', 'B-depart_time.period_of_day', 'B-depart_time.start_time', 'B-depart_time.time', 'B-depart_time.time_relative', 'B-economy', 'B-fare_amount', 'B-fare_basis_code', 'B-flight', 'B-flight_days', 'B-flight_mod', 'B-flight_number', 'B-flight_stop', 'B-flight_time', 'B-fromloc.airport_code', 'B-fromloc.airport_name', 'B-fromloc.city_name', 'B-fromloc.state_code', 'B-fromloc.state_name', 'B-meal', 'B-meal_code', 'B-meal_description', 'B-mod', 'B-month_name', 'B-or', 'B-period_of_day', 'B-restriction_code', 'B-return_date.date_relative', 'B-return_date.day_name', 'B-return_date.day_number', 'B-return_date.month_name', 'B-return_date.today_relative', 'B-return_time.period_mod', 'B-return_time.period_of_day', 'B-round_trip', 'B-state_code', 'B-state_name', 'B-stoploc.airport_code', 'B-stoploc.airport_name', 'B-stoploc.city_name', 'B-stoploc.state_code', 'B-time', 'B-time_relative', 'B-today_relative', 'B-toloc.airport_code', 'B-toloc.airport_name', 'B-toloc.city_name', 'B-toloc.country_name', 'B-toloc.state_code', 'B-toloc.state_name', 'B-transport_type', 'I-airline_name', 'I-airport_name', 'I-arrive_date.day_number', 'I-arrive_time.end_time', 'I-arrive_time.period_of_day', 'I-arrive_time.start_time', 'I-arrive_time.time', 'I-arrive_time.time_relative', 'I-city_name', 'I-class_type', 'I-cost_relative', 'I-depart_date.day_number', 'I-depart_date.today_relative', 'I-depart_time.end_time', 'I-depart_time.period_of_day', 'I-depart_time.start_time', 'I-depart_time.time', 'I-depart_time.time_relative', 'I-economy', 'I-fare_amount', 'I-fare_basis_code', 'I-flight_mod', 'I-flight_number', 'I-flight_stop', 'I-flight_time', 'I-fromloc.airport_name', 'I-fromloc.city_name', 'I-fromloc.state_name', 'I-meal_code', 'I-meal_description', 'I-restriction_code', 'I-return_date.date_relative', 'I-return_date.day_number', 'I-return_date.today_relative', 'I-round_trip', 'I-state_name', 'I-stoploc.city_name', 'I-time', 'I-today_relative', 'I-toloc.airport_name', 'I-toloc.city_name', 'I-toloc.state_name', 'I-transport_type', 'O']
ATIS_slot_effective_label = ['B-aircraft_code', 'B-airline_code', 'B-airline_name', 'B-airport_code', 'B-airport_name', 'B-arrive_date.date_relative', 'B-arrive_date.day_name', 'B-arrive_date.day_number', 'B-arrive_date.month_name', 'B-arrive_date.today_relative', 'B-arrive_time.end_time', 'B-arrive_time.period_mod', 'B-arrive_time.period_of_day', 'B-arrive_time.start_time', 'B-arrive_time.time', 'B-arrive_time.time_relative', 'B-booking_class', 'B-city_name', 'B-class_type', 'B-compartment', 'B-connect', 'B-cost_relative', 'B-day_name', 'B-day_number', 'B-days_code', 'B-depart_date.date_relative', 'B-depart_date.day_name', 'B-depart_date.day_number', 'B-depart_date.month_name', 'B-depart_date.today_relative', 'B-depart_date.year', 'B-depart_time.end_time', 'B-depart_time.period_mod', 'B-depart_time.period_of_day', 'B-depart_time.start_time', 'B-depart_time.time', 'B-depart_time.time_relative', 'B-economy', 'B-fare_amount', 'B-fare_basis_code', 'B-flight', 'B-flight_days', 'B-flight_mod', 'B-flight_number', 'B-flight_stop', 'B-flight_time', 'B-fromloc.airport_code', 'B-fromloc.airport_name', 'B-fromloc.city_name', 'B-fromloc.state_code', 'B-fromloc.state_name', 'B-meal', 'B-meal_code', 'B-meal_description', 'B-mod', 'B-month_name', 'B-or', 'B-period_of_day', 'B-restriction_code', 'B-return_date.date_relative', 'B-return_date.day_name', 'B-return_date.day_number', 'B-return_date.month_name', 'B-return_date.today_relative', 'B-return_time.period_mod', 'B-return_time.period_of_day', 'B-round_trip', 'B-state_code', 'B-state_name', 'B-stoploc.airport_code', 'B-stoploc.airport_name', 'B-stoploc.city_name', 'B-stoploc.state_code', 'B-time', 'B-time_relative', 'B-today_relative', 'B-toloc.airport_code', 'B-toloc.airport_name', 'B-toloc.city_name', 'B-toloc.country_name', 'B-toloc.state_code', 'B-toloc.state_name', 'B-transport_type', 'I-airline_name', 'I-airport_name', 'I-arrive_date.day_number', 'I-arrive_time.end_time', 'I-arrive_time.period_of_day', 'I-arrive_time.start_time', 'I-arrive_time.time', 'I-arrive_time.time_relative', 'I-city_name', 'I-class_type', 'I-cost_relative', 'I-depart_date.day_number', 'I-depart_date.today_relative', 'I-depart_time.end_time', 'I-depart_time.period_of_day', 'I-depart_time.start_time', 'I-depart_time.time', 'I-depart_time.time_relative', 'I-economy', 'I-fare_amount', 'I-fare_basis_code', 'I-flight_mod', 'I-flight_number', 'I-flight_stop', 'I-flight_time', 'I-fromloc.airport_name', 'I-fromloc.city_name', 'I-fromloc.state_name', 'I-meal_code', 'I-meal_description', 'I-restriction_code', 'I-return_date.date_relative', 'I-return_date.day_number', 'I-return_date.today_relative', 'I-round_trip', 'I-state_name', 'I-stoploc.city_name', 'I-time', 'I-today_relative', 'I-toloc.airport_name', 'I-toloc.city_name', 'I-toloc.state_name', 'I-transport_type', 'O']
ATIS_slot_effective_label2 = ['B-aircraft_code', 'B-airline_code', 'B-airline_name', 'B-airport_code', 'B-airport_name', 'B-arrive_date.date_relative', 'B-arrive_date.day_name', 'B-arrive_date.day_number', 'B-arrive_date.month_name', 'B-arrive_date.today_relative', 'B-arrive_time.end_time', 'B-arrive_time.period_mod', 'B-arrive_time.period_of_day', 'B-arrive_time.start_time', 'B-arrive_time.time', 'B-arrive_time.time_relative', 'B-booking_class', 'B-city_name', 'B-class_type', 'B-compartment', 'B-connect', 'B-cost_relative', 'B-day_name', 'B-day_number', 'B-days_code', 'B-depart_date.date_relative', 'B-depart_date.day_name', 'B-depart_date.day_number', 'B-depart_date.month_name', 'B-depart_date.today_relative', 'B-depart_date.year', 'B-depart_time.end_time', 'B-depart_time.period_mod', 'B-depart_time.period_of_day', 'B-depart_time.start_time', 'B-depart_time.time', 'B-depart_time.time_relative', 'B-economy', 'B-fare_amount', 'B-fare_basis_code', 'B-flight', 'B-flight_days', 'B-flight_mod', 'B-flight_number', 'B-flight_stop', 'B-flight_time', 'B-fromloc.airport_code', 'B-fromloc.airport_name', 'B-fromloc.city_name', 'B-fromloc.state_code', 'B-fromloc.state_name', 'B-meal', 'B-meal_code', 'B-meal_description', 'B-mod', 'B-month_name', 'B-or', 'B-period_of_day', 'B-restriction_code', 'B-return_date.date_relative', 'B-return_date.day_name', 'B-return_date.day_number', 'B-return_date.month_name', 'B-return_date.today_relative', 'B-return_time.period_mod', 'B-return_time.period_of_day', 'B-round_trip', 'B-state_code', 'B-state_name', 'B-stoploc.airport_code', 'B-stoploc.airport_name', 'B-stoploc.city_name', 'B-stoploc.state_code', 'B-time', 'B-time_relative', 'B-today_relative', 'B-toloc.airport_code', 'B-toloc.airport_name', 'B-toloc.city_name', 'B-toloc.country_name', 'B-toloc.state_code', 'B-toloc.state_name', 'B-transport_type', 'I-airline_name', 'I-airport_name', 'I-arrive_date.day_number', 'I-arrive_time.end_time', 'I-arrive_time.period_of_day', 'I-arrive_time.start_time', 'I-arrive_time.time', 'I-arrive_time.time_relative', 'I-city_name', 'I-class_type', 'I-cost_relative', 'I-depart_date.day_number', 'I-depart_date.today_relative', 'I-depart_time.end_time', 'I-depart_time.period_of_day', 'I-depart_time.start_time', 'I-depart_time.time', 'I-depart_time.time_relative', 'I-economy', 'I-fare_amount', 'I-fare_basis_code', 'I-flight_mod', 'I-flight_number', 'I-flight_stop', 'I-flight_time', 'I-fromloc.airport_name', 'I-fromloc.city_name', 'I-fromloc.state_name', 'I-meal_code', 'I-meal_description', 'I-restriction_code', 'I-return_date.date_relative', 'I-return_date.day_number', 'I-return_date.today_relative', 'I-round_trip', 'I-state_name', 'I-stoploc.city_name', 'I-time', 'I-today_relative', 'I-toloc.airport_name', 'I-toloc.city_name', 'I-toloc.state_name', 'I-transport_type']



with open(os.path.join("ATIS_solt", "seq.out")) as label_f:
    label_list = [label.replace("\n", "") for label in label_f.readlines()]
    label_list = [seq.split() for seq in label_list]
    #print(len(label_list), label_list)

with open(os.path.join("ATIS_solt", "label_test.txt")) as predict_f:
    predict_list = [predict_label.replace("\n", "") for predict_label in predict_f.readlines()]
    #print(len(predict_list), predict_list)
    predict_sentence_list = []
    for word in predict_list:
        if "[CLS]" == word:
            a_sentence = []
        a_sentence.append(word)
        if "[SEP]" == word:
            predict_sentence_list.append(a_sentence)
    #print(len(predict_sentence_list), predict_sentence_list)

y_test_list = []
clean_y_predict_list = []
assert len(label_list)==len(predict_sentence_list)
for y_test, y_predict in zip(label_list, predict_sentence_list):
    y_predict.remove('[CLS]')
    y_predict.remove('[SEP]')
    while '[Padding]' in y_predict:
        y_predict.remove('[Padding]')
    while '[##WordPiece]' in y_predict:
        y_predict.remove('[##WordPiece]')
    if len(y_predict)!=len(y_test):
        print(y_predict)
        print(y_test)
        print("~"*100)
    y_test_list.extend(y_test)
    clean_y_predict_list.extend(y_predict)

assert len(y_test_list)==len(clean_y_predict_list)



y_test_list, clean_y_predict_list = delete_both_sides_is_O_word(y_test_list, clean_y_predict_list)


show_metrics(y_test=y_test_list, y_predict=clean_y_predict_list, labels=ATIS_slot_effective_label2)

