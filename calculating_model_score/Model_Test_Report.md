## Some Task Output

以下结果，均为默认参数的结果，不一定是最优结果 num_train_epochs=3

[Task-Oriented-Dialogue-Dataset-Survey](https://github.com/AtmaHou/Task-Oriented-Dialogue-Dataset-Survey)

### Atis Intent Prediction
```
准确率: 0.9764837625979843
宏平均精确率: 0.7449854331023172
微平均精确率: 0.9764837625979843
加权平均精确率: 0.9750854106720163
宏平均召回率: 0.7499221418525216
微平均召回率: 0.9764837625979843
加权平均召回率: 0.9764837625979843
宏平均F1-score: 0.5650970439524852
微平均F1-score: 0.9764837625979843
加权平均F1-score: 0.9741996309183255
混淆矩阵输出:
 [[ 33   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0]
 [  0   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   1]
 [  0   0  48   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0]
 [  0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0]
 [  0   0   0   0  38   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0]
 [  0   0   0   0   0  18   0   0   0   0   0   0   0   0   0   0   0   0
    0   0]
 [  0   0   0   0   0   0  20   0   0   0   0   0   0   0   0   0   0   0
    0   1]
 [  2   0   0   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0
    0   0]
 [  0   0   0   0   0   0   0   0   0  10   0   0   0   0   0   0   0   0
    0   0]
 [  0   0   0   0   0   1   0   0   0   0 626   1   0   0   0   0   0   0
    0   4]
 [  0   0   2   0   0   0   0   0   0   0   4   6   0   0   0   0   0   0
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   8   0   0   0   0
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   7   0
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  36
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    6   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   3]]
分类报告:
                              precision    recall  f1-score   support

          atis_abbreviation       0.94      1.00      0.97        33
              atis_aircraft       1.00      0.89      0.94         9
               atis_airfare       0.94      1.00      0.97        48
   atis_airfare#atis_flight       0.00      0.00      0.00         1
               atis_airline       1.00      1.00      1.00        38
               atis_airport       0.95      1.00      0.97        18
              atis_capacity       1.00      0.95      0.98        21
                  atis_city       1.00      0.67      0.80         6
              atis_day_name       0.00      0.00      0.00         2
              atis_distance       1.00      1.00      1.00        10
                atis_flight       0.99      0.99      0.99       632
   atis_flight#atis_airfare       0.86      0.50      0.63        12
   atis_flight#atis_airline       0.00      0.00      0.00         1
             atis_flight_no       0.89      1.00      0.94         8
atis_flight_no#atis_airline       0.00      0.00      0.00         1
           atis_flight_time       1.00      1.00      1.00         1
           atis_ground_fare       1.00      1.00      1.00         7
        atis_ground_service       1.00      1.00      1.00        36
                  atis_meal       1.00      1.00      1.00         6
              atis_quantity       0.33      1.00      0.50         3

                avg / total       0.98      0.98      0.97       893
```

### Atis Slot Filling

这个值偏低是因为对类别处理与原论文不一致所致。
```
准确率: 0.9314658689148763
宏平均精确率: 0.6372126144323426
微平均精确率: 0.9314658689148763
加权平均精确率: 0.926265353242393
宏平均召回率: 0.6262272801046171
微平均召回率: 0.9314658689148763
加权平均召回率: 0.9314658689148763
宏平均F1-score: 0.4846269160280453
微平均F1-score: 0.935536738596012
加权平均F1-score: 0.922292687994663
混淆矩阵输出:
 [[ 26   3   0 ...   0   0   0]
 [  0  32   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ...   1   0   0]
 [  0   0   0 ...   0   0   1]
 [  0   4   0 ...   0   0   0]]
分类报告:
                               precision    recall  f1-score   support

             B-aircraft_code       1.00      0.79      0.88        33
              B-airline_code       0.58      0.94      0.72        34
              B-airline_name       0.99      1.00      1.00       101
              B-airport_code       1.00      0.22      0.36         9
              B-airport_name       1.00      0.05      0.09        21
 B-arrive_date.date_relative       0.00      0.00      0.00         2
      B-arrive_date.day_name       0.69      1.00      0.81        11
    B-arrive_date.day_number       1.00      0.83      0.91         6
    B-arrive_date.month_name       0.83      0.83      0.83         6
      B-arrive_time.end_time       0.00      0.00      0.00         8
 B-arrive_time.period_of_day       0.75      1.00      0.86         6
    B-arrive_time.start_time       0.00      0.00      0.00         8
          B-arrive_time.time       0.89      0.97      0.93        34
 B-arrive_time.time_relative       0.86      1.00      0.93        31
             B-booking_class       0.00      0.00      0.00         1
                 B-city_name       0.97      0.49      0.65        57
                B-class_type       0.96      1.00      0.98        24
               B-compartment       0.00      0.00      0.00         1
                   B-connect       1.00      1.00      1.00         6
             B-cost_relative       1.00      0.97      0.99        37
                  B-day_name       0.00      0.00      0.00         2
                 B-days_code       0.00      0.00      0.00         1
 B-depart_date.date_relative       0.81      1.00      0.89        17
      B-depart_date.day_name       0.99      0.98      0.98       212
    B-depart_date.day_number       0.98      1.00      0.99        55
    B-depart_date.month_name       0.98      0.98      0.98        56
B-depart_date.today_relative       1.00      1.00      1.00         9
          B-depart_date.year       1.00      1.00      1.00         3
      B-depart_time.end_time       0.38      1.00      0.55         3
    B-depart_time.period_mod       1.00      1.00      1.00         5
 B-depart_time.period_of_day       0.97      0.89      0.93       130
    B-depart_time.start_time       0.27      1.00      0.43         3
          B-depart_time.time       0.84      1.00      0.91        57
 B-depart_time.time_relative       0.98      1.00      0.99        65
                   B-economy       1.00      1.00      1.00         6
               B-fare_amount       1.00      1.00      1.00         2
           B-fare_basis_code       0.77      1.00      0.87        17
                    B-flight       0.00      0.00      0.00         1
               B-flight_days       1.00      1.00      1.00        10
                B-flight_mod       1.00      1.00      1.00        24
             B-flight_number       0.73      1.00      0.85        11
               B-flight_stop       1.00      1.00      1.00        21
               B-flight_time       1.00      1.00      1.00         1
      B-fromloc.airport_code       1.00      0.20      0.33         5
      B-fromloc.airport_name       0.35      1.00      0.52        12
         B-fromloc.city_name       0.98      1.00      0.99       704
        B-fromloc.state_code       0.96      1.00      0.98        23
        B-fromloc.state_name       0.94      0.94      0.94        17
                      B-meal       0.94      1.00      0.97        16
                 B-meal_code       0.00      0.00      0.00         1
          B-meal_description       1.00      0.90      0.95        10
                       B-mod       1.00      0.50      0.67         2
                        B-or       0.38      1.00      0.55         3
             B-period_of_day       0.00      0.00      0.00         4
          B-restriction_code       1.00      1.00      1.00         4
 B-return_date.date_relative       0.00      0.00      0.00         3
      B-return_date.day_name       0.00      0.00      0.00         2
                B-round_trip       1.00      0.97      0.99        73
                B-state_code       0.00      0.00      0.00         1
                B-state_name       0.00      0.00      0.00         9
      B-stoploc.airport_code       0.00      0.00      0.00         1
         B-stoploc.city_name       1.00      1.00      1.00        20
        B-toloc.airport_code       1.00      0.50      0.67         4
        B-toloc.airport_name       1.00      0.67      0.80         3
           B-toloc.city_name       0.97      1.00      0.98       716
        B-toloc.country_name       0.00      0.00      0.00         1
          B-toloc.state_code       0.95      1.00      0.97        18
          B-toloc.state_name       0.74      0.93      0.83        28
            B-transport_type       1.00      1.00      1.00        10
              I-airline_name       0.98      1.00      0.99        65
              I-airport_name       0.50      0.07      0.12        29
      I-arrive_time.end_time       0.00      0.00      0.00         8
    I-arrive_time.start_time       0.00      0.00      0.00         1
          I-arrive_time.time       0.77      0.97      0.86        35
 I-arrive_time.time_relative       0.00      0.00      0.00         4
                 I-city_name       0.83      0.33      0.48        30
                I-class_type       1.00      1.00      1.00        17
             I-cost_relative       1.00      0.67      0.80         3
    I-depart_date.day_number       1.00      1.00      1.00        15
      I-depart_time.end_time       0.67      0.67      0.67         3
 I-depart_time.period_of_day       0.00      0.00      0.00         1
    I-depart_time.start_time       0.00      0.00      0.00         1
          I-depart_time.time       0.93      1.00      0.96        52
 I-depart_time.time_relative       0.00      0.00      0.00         1
               I-fare_amount       1.00      1.00      1.00         2
                I-flight_mod       0.00      0.00      0.00         6
             I-flight_number       0.00      0.00      0.00         1
               I-flight_time       1.00      1.00      1.00         1
      I-fromloc.airport_name       0.33      1.00      0.50        15
         I-fromloc.city_name       0.96      1.00      0.98       177
        I-fromloc.state_name       0.00      0.00      0.00         1
          I-restriction_code       1.00      0.67      0.80         3
 I-return_date.date_relative       0.00      0.00      0.00         3
                I-round_trip       0.99      1.00      0.99        71
                I-state_name       0.00      0.00      0.00         1
         I-stoploc.city_name       1.00      1.00      1.00        10
        I-toloc.airport_name       1.00      0.33      0.50         3
           I-toloc.city_name       0.96      0.98      0.97       265
          I-toloc.state_name       1.00      1.00      1.00         1
            I-transport_type       0.00      0.00      0.00         1
                           O       0.00      0.00      0.00        14

                 avg / total       0.93      0.93      0.92      3677

```


### Snips Intent Prediction

```
准确率: 0.9842857142857143
宏平均精确率: 0.9858767424798239
微平均精确率: 0.9842857142857143
加权平均精确率: 0.9853440415050833
宏平均召回率: 0.9849107098074714
微平均召回率: 0.9842857142857143
加权平均召回率: 0.9842857142857143
宏平均F1-score: 0.9849281321280821
微平均F1-score: 0.9842857142857143
加权平均F1-score: 0.9843205635966433
混淆矩阵输出:
 [[124   0   0   0   0   0   0]
 [  0  92   0   0   0   0   0]
 [  0   2 102   0   0   0   0]
 [  0   0   0  85   0   1   0]
 [  0   0   0   0  80   0   0]
 [  0   0   0   0   0 107   0]
 [  0   0   0   0   0   8  99]]
分类报告:
                       precision    recall  f1-score   support

       AddToPlaylist       1.00      1.00      1.00       124
      BookRestaurant       0.98      1.00      0.99        92
          GetWeather       1.00      0.98      0.99       104
           PlayMusic       1.00      0.99      0.99        86
            RateBook       1.00      1.00      1.00        80
  SearchCreativeWork       0.92      1.00      0.96       107
SearchScreeningEvent       1.00      0.93      0.96       107

         avg / total       0.99      0.98      0.98       700
```

### Snips Slot Filling
```
准确率: 0.9457740078764011
宏平均精确率: 0.8909580647773255
微平均精确率: 0.9457740078764011
加权平均精确率: 0.9470905924923945
宏平均召回率: 0.885268678299942
微平均召回率: 0.9457740078764011
加权平均召回率: 0.9457740078764011
宏平均F1-score: 0.8614473609012004
微平均F1-score: 0.9457740078764011
加权平均F1-score: 0.9456687237548242
混淆矩阵输出:
 [[  2   1   0 ...   0   0   1]
 [  0 105   0 ...   0   0   0]
 [  0   0  43 ...   0   0   0]
 ...
 [  0   0   0 ... 144   0   0]
 [  0   0   0 ...   0  10   0]
 [  1   1   0 ...   0   0   0]]
分类报告:
                               precision    recall  f1-score   support

                     B-album       0.33      0.20      0.25        10
                    B-artist       0.91      0.98      0.95       107
               B-best_rating       1.00      1.00      1.00        43
                      B-city       0.98      0.98      0.98        60
     B-condition_description       1.00      1.00      1.00        28
     B-condition_temperature       1.00      1.00      1.00        23
                   B-country       1.00      0.98      0.99        44
                   B-cuisine       0.85      0.79      0.81        14
          B-current_location       1.00      1.00      1.00        14
               B-entity_name       0.93      0.79      0.85        33
                  B-facility       1.00      1.00      1.00         3
                     B-genre       1.00      1.00      1.00         5
            B-geographic_poi       1.00      1.00      1.00        11
             B-location_name       1.00      0.96      0.98        24
                B-movie_name       1.00      0.85      0.92        47
                B-movie_type       1.00      1.00      1.00        33
                B-music_item       0.97      0.99      0.98       104
      B-object_location_type       1.00      1.00      1.00        22
               B-object_name       0.90      0.93      0.91       147
B-object_part_of_series_type       0.85      1.00      0.92        11
             B-object_select       0.95      1.00      0.98        40
               B-object_type       0.98      0.98      0.98       162
    B-party_size_description       1.00      1.00      1.00        10
         B-party_size_number       1.00      1.00      1.00        50
                  B-playlist       0.95      0.95      0.95       129
            B-playlist_owner       1.00      0.99      0.99        70
                       B-poi       1.00      0.62      0.77         8
               B-rating_unit       1.00      1.00      1.00        40
              B-rating_value       1.00      1.00      1.00        80
           B-restaurant_name       0.79      1.00      0.88        15
           B-restaurant_type       1.00      0.97      0.98        65
               B-served_dish       0.75      0.75      0.75        12
                   B-service       1.00      0.96      0.98        24
                      B-sort       0.91      0.97      0.94        32
          B-spatial_relation       0.99      1.00      0.99        71
                     B-state       0.98      0.98      0.98        59
                 B-timeRange       0.98      0.99      0.99       107
                     B-track       0.50      0.56      0.53         9
                      B-year       1.00      1.00      1.00        24
                     I-album       0.30      0.38      0.33        21
                    I-artist       0.96      0.98      0.97       112
                      I-city       1.00      1.00      1.00        19
                   I-country       1.00      0.92      0.96        25
                   I-cuisine       0.00      0.00      0.00         1
          I-current_location       1.00      1.00      1.00         7
               I-entity_name       0.96      0.93      0.94        54
                     I-genre       1.00      1.00      1.00         2
            I-geographic_poi       1.00      1.00      1.00        33
             I-location_name       1.00      0.97      0.98        30
                I-movie_name       0.96      0.86      0.91       121
                I-movie_type       1.00      1.00      1.00        16
                I-music_item       1.00      1.00      1.00         5
      I-object_location_type       1.00      1.00      1.00        16
               I-object_name       0.94      0.96      0.95       399
I-object_part_of_series_type       0.00      0.00      0.00         1
               I-object_type       1.00      0.98      0.99        66
    I-party_size_description       1.00      1.00      1.00        35
                  I-playlist       0.97      0.94      0.95       231
            I-playlist_owner       1.00      1.00      1.00         7
                       I-poi       1.00      0.91      0.95        11
           I-restaurant_name       0.90      1.00      0.95        36
           I-restaurant_type       1.00      1.00      1.00         7
               I-served_dish       0.67      0.50      0.57         4
                   I-service       1.00      1.00      1.00         5
                      I-sort       1.00      1.00      1.00         9
          I-spatial_relation       0.98      1.00      0.99        42
                     I-state       0.75      1.00      0.86         6
                 I-timeRange       0.99      1.00      1.00       144
                     I-track       0.50      0.48      0.49        21
                           O       0.00      0.00      0.00        25

                 avg / total       0.95      0.95      0.95      3301
```

## Snips Slot Filling and Intent Prediction

Intent Prediction

```
准确率: 0.9814285714285714
宏平均精确率: 0.9827947881945077
微平均精确率: 0.9814285714285714
加权平均精确率: 0.9825026562964853
宏平均召回率: 0.9826050118106192
微平均召回率: 0.9814285714285714
加权平均召回率: 0.9814285714285714
宏平均F1-score: 0.9821708231356349
微平均F1-score: 0.9814285714285714
加权平均F1-score: 0.9814041423909138
混淆矩阵输出:
 [[124   0   0   0   0   0   0]
 [  0  92   0   0   0   0   0]
 [  0   1 103   0   0   0   0]
 [  0   0   0  86   0   0   0]
 [  0   0   0   0  80   0   0]
 [  0   0   0   2   0 105   0]
 [  0   0   0   0   0  10  97]]
分类报告:
                       precision    recall  f1-score   support

       AddToPlaylist       1.00      1.00      1.00       124
      BookRestaurant       0.99      1.00      0.99        92
          GetWeather       1.00      0.99      1.00       104
           PlayMusic       0.98      1.00      0.99        86
            RateBook       1.00      1.00      1.00        80
  SearchCreativeWork       0.91      0.98      0.95       107
SearchScreeningEvent       1.00      0.91      0.95       107

         avg / total       0.98      0.98      0.98       700
```

Slot Filling

```
宏平均精确率: 0.9117980613400293
微平均精确率: 0.9554545454545454
加权平均精确率: 0.9568517403727163
宏平均召回率: 0.9204978736517972
微平均召回率: 0.9554545454545454
加权平均召回率: 0.9554545454545454
宏平均F1-score: 0.8871082437233873
微平均F1-score: 0.9554545454545454
加权平均F1-score: 0.9548568099716878
混淆矩阵输出:
 [[  4   0   0 ...   0   0   1]
 [  0 103   0 ...   0   0   0]
 [  0   0  43 ...   0   0   0]
 ...
 [  0   0   0 ... 143   0   0]
 [  0   0   0 ...   0  19   0]
 [  1   1   0 ...   1   1   0]]
分类报告:
                               precision    recall  f1-score   support

                     B-album       0.57      0.40      0.47        10
                    B-artist       0.96      0.96      0.96       107
               B-best_rating       1.00      1.00      1.00        43
                      B-city       1.00      0.97      0.98        60
     B-condition_description       1.00      1.00      1.00        28
     B-condition_temperature       1.00      1.00      1.00        23
                   B-country       0.98      0.98      0.98        44
                   B-cuisine       0.87      0.93      0.90        14
          B-current_location       1.00      1.00      1.00        14
               B-entity_name       0.88      0.85      0.86        33
                  B-facility       1.00      1.00      1.00         3
                     B-genre       0.71      1.00      0.83         5
            B-geographic_poi       1.00      1.00      1.00        11
             B-location_name       1.00      0.96      0.98        24
                B-movie_name       1.00      0.85      0.92        47
                B-movie_type       1.00      1.00      1.00        33
                B-music_item       0.98      1.00      0.99       104
      B-object_location_type       1.00      1.00      1.00        22
               B-object_name       0.89      0.93      0.91       147
B-object_part_of_series_type       0.92      1.00      0.96        11
             B-object_select       1.00      1.00      1.00        40
               B-object_type       0.99      0.99      0.99       162
    B-party_size_description       1.00      1.00      1.00        10
         B-party_size_number       0.98      1.00      0.99        50
                  B-playlist       0.95      0.97      0.96       129
            B-playlist_owner       1.00      1.00      1.00        70
                       B-poi       0.80      1.00      0.89         8
               B-rating_unit       1.00      1.00      1.00        40
              B-rating_value       1.00      1.00      1.00        80
           B-restaurant_name       0.94      1.00      0.97        15
           B-restaurant_type       1.00      0.97      0.98        65
               B-served_dish       0.90      0.75      0.82        12
                   B-service       1.00      1.00      1.00        24
                      B-sort       0.97      1.00      0.98        32
          B-spatial_relation       0.96      1.00      0.98        71
                     B-state       0.98      0.98      0.98        59
                 B-timeRange       0.98      0.97      0.98       107
                     B-track       0.50      0.67      0.57         9
                      B-year       1.00      1.00      1.00        24
                     I-album       0.67      0.29      0.40        21
                    I-artist       0.97      0.99      0.98       112
                      I-city       1.00      1.00      1.00        19
                   I-country       1.00      0.96      0.98        25
                   I-cuisine       1.00      1.00      1.00         1
          I-current_location       1.00      1.00      1.00         7
               I-entity_name       0.93      0.93      0.93        54
                     I-genre       0.67      1.00      0.80         2
            I-geographic_poi       1.00      1.00      1.00        33
             I-location_name       1.00      0.97      0.98        30
                I-movie_name       0.98      0.87      0.92       121
                I-movie_type       1.00      1.00      1.00        16
                I-music_item       1.00      1.00      1.00         5
      I-object_location_type       1.00      1.00      1.00        16
               I-object_name       0.96      0.97      0.96       399
I-object_part_of_series_type       0.00      0.00      0.00         1
               I-object_type       1.00      1.00      1.00        66
    I-party_size_description       1.00      1.00      1.00        35
                  I-playlist       0.97      0.95      0.96       231
            I-playlist_owner       1.00      1.00      1.00         7
                       I-poi       1.00      1.00      1.00        11
           I-restaurant_name       0.95      1.00      0.97        36
           I-restaurant_type       1.00      1.00      1.00         7
               I-served_dish       0.67      0.50      0.57         4
                   I-service       1.00      1.00      1.00         5
                      I-sort       1.00      1.00      1.00         9
          I-spatial_relation       0.98      1.00      0.99        42
                     I-state       0.86      1.00      0.92         6
                 I-timeRange       0.99      0.99      0.99       144
                     I-track       0.54      0.90      0.68        21
                           O       0.00      0.00      0.00        24

                 avg / total       0.96      0.96      0.95      3300
```
