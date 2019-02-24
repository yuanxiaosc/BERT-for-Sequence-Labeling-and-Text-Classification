import os
from sklearn_metrics_function import show_metrics,delete_both_sides_is_O_word

SNIPS_slot_label = ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', 'B-album', 'B-artist', 'B-best_rating', 'B-city', 'B-condition_description', 'B-condition_temperature', 'B-country', 'B-cuisine', 'B-current_location', 'B-entity_name', 'B-facility', 'B-genre', 'B-geographic_poi', 'B-location_name', 'B-movie_name', 'B-movie_type', 'B-music_item', 'B-object_location_type', 'B-object_name', 'B-object_part_of_series_type', 'B-object_select', 'B-object_type', 'B-party_size_description', 'B-party_size_number', 'B-playlist', 'B-playlist_owner', 'B-poi', 'B-rating_unit', 'B-rating_value', 'B-restaurant_name', 'B-restaurant_type', 'B-served_dish', 'B-service', 'B-sort', 'B-spatial_relation', 'B-state', 'B-timeRange', 'B-track', 'B-year', 'I-album', 'I-artist', 'I-city', 'I-country', 'I-cuisine', 'I-current_location', 'I-entity_name', 'I-facility', 'I-genre', 'I-geographic_poi', 'I-location_name', 'I-movie_name', 'I-movie_type', 'I-music_item', 'I-object_location_type', 'I-object_name', 'I-object_part_of_series_type', 'I-object_select', 'I-object_type', 'I-party_size_description', 'I-playlist', 'I-playlist_owner', 'I-poi', 'I-restaurant_name', 'I-restaurant_type', 'I-served_dish', 'I-service', 'I-sort', 'I-spatial_relation', 'I-state', 'I-timeRange', 'I-track', 'O']
SNIPS_slot_effective_label = ['B-album', 'B-artist', 'B-best_rating', 'B-city', 'B-condition_description', 'B-condition_temperature', 'B-country', 'B-cuisine', 'B-current_location', 'B-entity_name', 'B-facility', 'B-genre', 'B-geographic_poi', 'B-location_name', 'B-movie_name', 'B-movie_type', 'B-music_item', 'B-object_location_type', 'B-object_name', 'B-object_part_of_series_type', 'B-object_select', 'B-object_type', 'B-party_size_description', 'B-party_size_number', 'B-playlist', 'B-playlist_owner', 'B-poi', 'B-rating_unit', 'B-rating_value', 'B-restaurant_name', 'B-restaurant_type', 'B-served_dish', 'B-service', 'B-sort', 'B-spatial_relation', 'B-state', 'B-timeRange', 'B-track', 'B-year', 'I-album', 'I-artist', 'I-city', 'I-country', 'I-cuisine', 'I-current_location', 'I-entity_name', 'I-facility', 'I-genre', 'I-geographic_poi', 'I-location_name', 'I-movie_name', 'I-movie_type', 'I-music_item', 'I-object_location_type', 'I-object_name', 'I-object_part_of_series_type', 'I-object_select', 'I-object_type', 'I-party_size_description', 'I-playlist', 'I-playlist_owner', 'I-poi', 'I-restaurant_name', 'I-restaurant_type', 'I-served_dish', 'I-service', 'I-sort', 'I-spatial_relation', 'I-state', 'I-timeRange', 'I-track', 'O']
SNIPS_slot_effective_label2 = ['B-album', 'B-artist', 'B-best_rating', 'B-city', 'B-condition_description', 'B-condition_temperature', 'B-country', 'B-cuisine', 'B-current_location', 'B-entity_name', 'B-facility', 'B-genre', 'B-geographic_poi', 'B-location_name', 'B-movie_name', 'B-movie_type', 'B-music_item', 'B-object_location_type', 'B-object_name', 'B-object_part_of_series_type', 'B-object_select', 'B-object_type', 'B-party_size_description', 'B-party_size_number', 'B-playlist', 'B-playlist_owner', 'B-poi', 'B-rating_unit', 'B-rating_value', 'B-restaurant_name', 'B-restaurant_type', 'B-served_dish', 'B-service', 'B-sort', 'B-spatial_relation', 'B-state', 'B-timeRange', 'B-track', 'B-year', 'I-album', 'I-artist', 'I-city', 'I-country', 'I-cuisine', 'I-current_location', 'I-entity_name', 'I-facility', 'I-genre', 'I-geographic_poi', 'I-location_name', 'I-movie_name', 'I-movie_type', 'I-music_item', 'I-object_location_type', 'I-object_name', 'I-object_part_of_series_type', 'I-object_select', 'I-object_type', 'I-party_size_description', 'I-playlist', 'I-playlist_owner', 'I-poi', 'I-restaurant_name', 'I-restaurant_type', 'I-served_dish', 'I-service', 'I-sort', 'I-spatial_relation', 'I-state', 'I-timeRange', 'I-track']



with open(os.path.join("SNIPS_slot", "seq.out")) as label_f:
    label_list = [label.replace("\n", "") for label in label_f.readlines()]
    label_list = [seq.split() for seq in label_list]
    #print(len(label_list), label_list)

with open(os.path.join("SNIPS_slot", "label_test.txt")) as predict_f:
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

show_metrics(y_test=y_test_list, y_predict=clean_y_predict_list, labels=SNIPS_slot_effective_label)

