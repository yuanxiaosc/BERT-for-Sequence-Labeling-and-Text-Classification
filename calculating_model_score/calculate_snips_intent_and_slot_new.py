import os
import numpy as np
from sklearn_metrics_function import show_metrics,delete_both_sides_is_O_word

print("-"*100)
print("Slot Intent Task Report")
print("-"*100)
SNIPS_intent_label = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

with open(os.path.join("join_task", "label")) as label_f:
    intent_label_list = [label.replace("\n", "") for label in label_f.readlines()]

with open(os.path.join("join_task", "intent_prediction_test_results.txt")) as intent_f:
    predict_intent_label_list = [label.replace("\n", "") for label in intent_f.readlines()]

assert len(intent_label_list)==len(predict_intent_label_list)

show_metrics(y_test=intent_label_list, y_predict=predict_intent_label_list, labels=SNIPS_intent_label)

print("-"*100)
print("Slot Filling Task Report")
print("-"*100)

SNIPS_slot_label = ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', 'B-album', 'B-artist', 'B-best_rating', 'B-city', 'B-condition_description', 'B-condition_temperature', 'B-country', 'B-cuisine', 'B-current_location', 'B-entity_name', 'B-facility', 'B-genre', 'B-geographic_poi', 'B-location_name', 'B-movie_name', 'B-movie_type', 'B-music_item', 'B-object_location_type', 'B-object_name', 'B-object_part_of_series_type', 'B-object_select', 'B-object_type', 'B-party_size_description', 'B-party_size_number', 'B-playlist', 'B-playlist_owner', 'B-poi', 'B-rating_unit', 'B-rating_value', 'B-restaurant_name', 'B-restaurant_type', 'B-served_dish', 'B-service', 'B-sort', 'B-spatial_relation', 'B-state', 'B-timeRange', 'B-track', 'B-year', 'I-album', 'I-artist', 'I-city', 'I-country', 'I-cuisine', 'I-current_location', 'I-entity_name', 'I-facility', 'I-genre', 'I-geographic_poi', 'I-location_name', 'I-movie_name', 'I-movie_type', 'I-music_item', 'I-object_location_type', 'I-object_name', 'I-object_part_of_series_type', 'I-object_select', 'I-object_type', 'I-party_size_description', 'I-playlist', 'I-playlist_owner', 'I-poi', 'I-restaurant_name', 'I-restaurant_type', 'I-served_dish', 'I-service', 'I-sort', 'I-spatial_relation', 'I-state', 'I-timeRange', 'I-track', 'O']
SNIPS_slot_effective_label = ['B-album', 'B-artist', 'B-best_rating', 'B-city', 'B-condition_description', 'B-condition_temperature', 'B-country', 'B-cuisine', 'B-current_location', 'B-entity_name', 'B-facility', 'B-genre', 'B-geographic_poi', 'B-location_name', 'B-movie_name', 'B-movie_type', 'B-music_item', 'B-object_location_type', 'B-object_name', 'B-object_part_of_series_type', 'B-object_select', 'B-object_type', 'B-party_size_description', 'B-party_size_number', 'B-playlist', 'B-playlist_owner', 'B-poi', 'B-rating_unit', 'B-rating_value', 'B-restaurant_name', 'B-restaurant_type', 'B-served_dish', 'B-service', 'B-sort', 'B-spatial_relation', 'B-state', 'B-timeRange', 'B-track', 'B-year', 'I-album', 'I-artist', 'I-city', 'I-country', 'I-cuisine', 'I-current_location', 'I-entity_name', 'I-facility', 'I-genre', 'I-geographic_poi', 'I-location_name', 'I-movie_name', 'I-movie_type', 'I-music_item', 'I-object_location_type', 'I-object_name', 'I-object_part_of_series_type', 'I-object_select', 'I-object_type', 'I-party_size_description', 'I-playlist', 'I-playlist_owner', 'I-poi', 'I-restaurant_name', 'I-restaurant_type', 'I-served_dish', 'I-service', 'I-sort', 'I-spatial_relation', 'I-state', 'I-timeRange', 'I-track', 'O']
SNIPS_slot_effective_label2 = ['B-album', 'B-artist', 'B-best_rating', 'B-city', 'B-condition_description', 'B-condition_temperature', 'B-country', 'B-cuisine', 'B-current_location', 'B-entity_name', 'B-facility', 'B-genre', 'B-geographic_poi', 'B-location_name', 'B-movie_name', 'B-movie_type', 'B-music_item', 'B-object_location_type', 'B-object_name', 'B-object_part_of_series_type', 'B-object_select', 'B-object_type', 'B-party_size_description', 'B-party_size_number', 'B-playlist', 'B-playlist_owner', 'B-poi', 'B-rating_unit', 'B-rating_value', 'B-restaurant_name', 'B-restaurant_type', 'B-served_dish', 'B-service', 'B-sort', 'B-spatial_relation', 'B-state', 'B-timeRange', 'B-track', 'B-year', 'I-album', 'I-artist', 'I-city', 'I-country', 'I-cuisine', 'I-current_location', 'I-entity_name', 'I-facility', 'I-genre', 'I-geographic_poi', 'I-location_name', 'I-movie_name', 'I-movie_type', 'I-music_item', 'I-object_location_type', 'I-object_name', 'I-object_part_of_series_type', 'I-object_select', 'I-object_type', 'I-party_size_description', 'I-playlist', 'I-playlist_owner', 'I-poi', 'I-restaurant_name', 'I-restaurant_type', 'I-served_dish', 'I-service', 'I-sort', 'I-spatial_relation', 'I-state', 'I-timeRange', 'I-track']

with open(os.path.join("join_task", "seq_out")) as slot_f:
    slot_label_list = [label.split() for label in slot_f.readlines()]

with open(os.path.join("join_task", "slot_filling_test_results.txt")) as slot_predict_f:
    predict_slot_sentence_list = [predict_label.split() for predict_label in slot_predict_f.readlines()]

assert len(slot_label_list)==len(predict_slot_sentence_list)

slot_test_list = []
clean_predict_slot_list = []

seqence_length_dont_match_index = 0
for y_test, y_predict in zip(slot_label_list, predict_slot_sentence_list):
    y_predict.remove('[CLS]')
    y_predict.remove('[SEP]')
    while '[Padding]' in y_predict:
        y_predict.remove('[Padding]')
    while '[##WordPiece]' in y_predict:
        y_predict.remove('[##WordPiece]')
    if len(y_predict) > len(y_test):
        print(seqence_length_dont_match_index)
        print(y_predict)
        print(y_test)
        print("~"*100)
        y_predict = y_predict[0:len(y_test)]
    elif len(y_predict) < len(y_test):
        print(seqence_length_dont_match_index)
        print(y_predict)
        print(y_test)
        print("~"*100)
        y_predict = y_predict + ["O"] * (len(y_test)-len(y_predict))
    assert len(y_predict)==len(y_test)
    slot_test_list.extend(y_test)
    clean_predict_slot_list.extend(y_predict)
    seqence_length_dont_match_index +=1

assert len(slot_test_list) == len(clean_predict_slot_list)

slot_test_list, clean_predict_slot_list = delete_both_sides_is_O_word(slot_test_list, clean_predict_slot_list)

show_metrics(y_test=slot_test_list, y_predict=clean_predict_slot_list, labels=SNIPS_slot_effective_label)