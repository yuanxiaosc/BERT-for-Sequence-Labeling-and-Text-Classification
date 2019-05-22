from sklearn import metrics
import os
import sys
import time
#import numpy
#numpy.set_printoptions(threshold=numpy.nan)


class Logger(object):
    """store log to txt file"""
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class Sequence_Labeling_and_Text_Classification_Calculate(object):

    def get_slot_labels(self):
        """for Sequence_Labeling labels"""
        raise NotImplementedError()

    def get_intent_labels(self):
        """for Text_Classification labels"""
        raise NotImplementedError()


    @classmethod
    def show_metrics(cls, y_test_list, y_predict_list, label_list):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        print('准确率:', metrics.accuracy_score(y_test_list, y_predict_list))  # 预测准确率输出

        print('宏平均精确率:', metrics.precision_score(y_test_list, y_predict_list, average='macro'))  # 预测宏平均精确率输出
        print('微平均精确率:', metrics.precision_score(y_test_list, y_predict_list, average='micro'))  # 预测微平均精确率输出
        print('加权平均精确率:', metrics.precision_score(y_test_list, y_predict_list, average='weighted'))  # 预测加权平均精确率输出

        print('宏平均召回率:', metrics.recall_score(y_test_list, y_predict_list, average='macro'))  # 预测宏平均召回率输出
        print('微平均召回率:', metrics.recall_score(y_test_list, y_predict_list, average='micro'))  # 预测微平均召回率输出
        print('加权平均召回率:', metrics.recall_score(y_test_list, y_predict_list, average='micro'))  # 预测加权平均召回率输出

        print('宏平均F1-score:', metrics.f1_score(y_test_list, y_predict_list, labels=label_list, average='macro'))  # 预测宏平均f1-score输出
        print('微平均F1-score:', metrics.f1_score(y_test_list, y_predict_list, labels=label_list, average='micro'))  # 预测微平均f1-score输出
        print('加权平均F1-score:',
              metrics.f1_score(y_test_list, y_predict_list, labels=label_list, average='weighted'))  # 预测加权平均f1-score输出
        a_confusion_matrix = metrics.confusion_matrix(y_test_list, y_predict_list)

        print('混淆矩阵输出:\n', a_confusion_matrix)  # 混淆矩阵输出
        print('分类报告:\n', metrics.classification_report(y_test_list, y_predict_list))  # 分类报告输出
        print("\n")

    @classmethod
    def delete_both_sides_is_O_word(cls, y_test_list, clean_y_predict_list):
        new_y_test_list, new_clean_y_predict_list = [], []
        for test, pred in zip(y_test_list, clean_y_predict_list):
            if test == "O" and pred == "O":
                continue
            new_y_test_list.append(test)
            new_clean_y_predict_list.append(pred)
        assert len(new_y_test_list) == len(new_clean_y_predict_list)
        return new_y_test_list, new_clean_y_predict_list

class Snips_Slot_Filling_and_Intent_Detection_Calculate(Sequence_Labeling_and_Text_Classification_Calculate):

    def __init__(self, path_to_label_file=None, path_to_predict_label_file=None, log_out_file=None):
        if path_to_label_file is None and path_to_predict_label_file is None:
            raise Exception("At least have `path_to_label_file")
        self.path_to_label_file = path_to_label_file
        if path_to_predict_label_file is not None:
            self.path_to_predict_label_file = path_to_predict_label_file
        else:
            self.path_to_predict_label_file = path_to_label_file
        if log_out_file is None:
            self.log_out_file = os.getcwd()
        else:
            if not os.path.exists(log_out_file):
                os.makedirs(log_out_file)
            self.log_out_file = log_out_file

    def get_intent_label_list(self, path_to_intent_label_file):
        with open(path_to_intent_label_file) as label_f:
            intent_label_list = [label.replace("\n", "") for label in label_f.readlines()]
        return intent_label_list

    def get_predict_intent_label_list(self, path_to_predict_intent_label_file):
        with open(path_to_predict_intent_label_file) as intent_f:
            predict_intent_label_list = [label.replace("\n", "") for label in intent_f.readlines()]
        return predict_intent_label_list

    def _get_slot_sententce_list(self, path_to_slot_sentence_file):
        with open(path_to_slot_sentence_file) as slot_f:
            slot_sententce_list = [sententce.split() for sententce in slot_f.readlines()]
        return slot_sententce_list

    def _get_predict_slot_sentence_list(self, path_to_slot_filling_test_results_file):
        with open(path_to_slot_filling_test_results_file) as slot_predict_f:
            predict_slot_sentence_list = [predict_label.split() for predict_label in slot_predict_f.readlines()]
        return predict_slot_sentence_list

    def producte_slot_list(self):
        """input seq.out and slot_filling_test_results.txt file
           output slot_test_list, clean_predict_slot_list
        """
        path_to_slot_sentence_file = os.path.join(self.path_to_label_file, "seq.out")
        slot_sententce_list = self._get_predict_slot_sentence_list(path_to_slot_sentence_file)
        path_to_slot_filling_test_results_file = os.path.join(self.path_to_predict_label_file, "slot_filling_test_results.txt")
        predict_slot_sentence_list = self._get_predict_slot_sentence_list(path_to_slot_filling_test_results_file)
        slot_test_list = []
        clean_predict_slot_list = []
        seqence_length_dont_match_index = 0
        for y_test, y_predict in zip(slot_sententce_list, predict_slot_sentence_list):
            y_predict.remove('[CLS]')
            y_predict.remove('[SEP]')
            while '[Padding]' in y_predict:
                y_predict.remove('[Padding]')
            while '[##WordPiece]' in y_predict:
                y_predict.remove('[##WordPiece]')
            if len(y_predict) > len(y_test):
                #print(seqence_length_dont_match_index)
                #print(y_predict)
                #print(y_test)
                #print("~" * 100)
                seqence_length_dont_match_index += 1
                y_predict = y_predict[0:len(y_test)]
            elif len(y_predict) < len(y_test):
                #print(seqence_length_dont_match_index)
                #print(y_predict)
                #print(y_test)
                #print("~" * 100)
                y_predict = y_predict + ["O"] * (len(y_test) - len(y_predict))
                seqence_length_dont_match_index += 1
            assert len(y_predict) == len(y_test)
            slot_test_list.extend(y_test)
            clean_predict_slot_list.extend(y_predict)
        #print("seqence_length_dont_match numbers", seqence_length_dont_match_index)
        return slot_test_list, clean_predict_slot_list

    def get_slot_model_labels(self):
        """contain ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', 'O'] + Task labels"""
        return ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', 'B-album', 'B-artist', 'B-best_rating', 'B-city',
                'B-condition_description', 'B-condition_temperature', 'B-country', 'B-cuisine', 'B-current_location',
                'B-entity_name', 'B-facility', 'B-genre', 'B-geographic_poi', 'B-location_name', 'B-movie_name',
                'B-movie_type', 'B-music_item', 'B-object_location_type', 'B-object_name',
                'B-object_part_of_series_type', 'B-object_select', 'B-object_type', 'B-party_size_description',
                'B-party_size_number', 'B-playlist', 'B-playlist_owner', 'B-poi', 'B-rating_unit', 'B-rating_value',
                'B-restaurant_name', 'B-restaurant_type', 'B-served_dish', 'B-service', 'B-sort', 'B-spatial_relation',
                'B-state', 'B-timeRange', 'B-track', 'B-year', 'I-album', 'I-artist', 'I-city', 'I-country',
                'I-cuisine', 'I-current_location', 'I-entity_name', 'I-facility', 'I-genre', 'I-geographic_poi',
                'I-location_name', 'I-movie_name', 'I-movie_type', 'I-music_item', 'I-object_location_type',
                'I-object_name', 'I-object_part_of_series_type', 'I-object_select', 'I-object_type',
                'I-party_size_description', 'I-playlist', 'I-playlist_owner', 'I-poi', 'I-restaurant_name',
                'I-restaurant_type', 'I-served_dish', 'I-service', 'I-sort', 'I-spatial_relation', 'I-state',
                'I-timeRange', 'I-track', 'O']

    def get_slot_labels(self):
        """only contain Task labels"""
        return ['B-album', 'B-artist', 'B-best_rating', 'B-city',
                'B-condition_description', 'B-condition_temperature', 'B-country', 'B-cuisine', 'B-current_location',
                'B-entity_name', 'B-facility', 'B-genre', 'B-geographic_poi', 'B-location_name', 'B-movie_name',
                'B-movie_type', 'B-music_item', 'B-object_location_type', 'B-object_name',
                'B-object_part_of_series_type', 'B-object_select', 'B-object_type', 'B-party_size_description',
                'B-party_size_number', 'B-playlist', 'B-playlist_owner', 'B-poi', 'B-rating_unit', 'B-rating_value',
                'B-restaurant_name', 'B-restaurant_type', 'B-served_dish', 'B-service', 'B-sort', 'B-spatial_relation',
                'B-state', 'B-timeRange', 'B-track', 'B-year', 'I-album', 'I-artist', 'I-city', 'I-country',
                'I-cuisine', 'I-current_location', 'I-entity_name', 'I-facility', 'I-genre', 'I-geographic_poi',
                'I-location_name', 'I-movie_name', 'I-movie_type', 'I-music_item', 'I-object_location_type',
                'I-object_name', 'I-object_part_of_series_type', 'I-object_select', 'I-object_type',
                'I-party_size_description', 'I-playlist', 'I-playlist_owner', 'I-poi', 'I-restaurant_name',
                'I-restaurant_type', 'I-served_dish', 'I-service', 'I-sort', 'I-spatial_relation', 'I-state',
                'I-timeRange', 'I-track']

    def get_intent_labels(self):
        return ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']


    def show_intent_prediction_report(self):
        sys.stdout = Logger(os.path.join(self.log_out_file, "log.txt"))
        path_to_intent_label_lfile = os.path.join(self.path_to_label_file, "label")
        intent_label_list = self.get_intent_label_list(path_to_intent_label_lfile)
        path_to_predict_intent_label_file = os.path.join(self.path_to_predict_label_file, "intent_prediction_test_results.txt")
        predict_intent_label_list = self.get_predict_intent_label_list(path_to_predict_intent_label_file)
        labels = self.get_intent_labels()
        print("---show_intent_prediction_report---")
        self.show_metrics(intent_label_list, predict_intent_label_list, labels)
        print("--"*30)

    def show_slot_filling_report(self):
        sys.stdout = Logger(os.path.join(self.log_out_file, "log.txt"))
        slot_test_list, clean_predict_slot_list = self.producte_slot_list()
        slot_test_list, clean_predict_slot_list = self.delete_both_sides_is_O_word(slot_test_list, clean_predict_slot_list)
        labels = self.get_slot_labels()
        print("---show_slot_filling_report---")
        self.show_metrics(slot_test_list, clean_predict_slot_list, labels)
        print("--"*30)


if __name__=='__main__':
    path_to_label_file = "/home/b418/jupyter_workspace/yuanxiao/" \
                         "BERT-for-Sequence-Labeling-and-Text-Classification/" \
                         "data/snips_Intent_Detection_and_Slot_Filling/test/"

    path_to_predict_label_file = "snips_join_task_epoch10_test4088ckpt"
    log_out_file = "snips_join_task_epoch10_test4088ckpt"
    intent_slot_reports = Snips_Slot_Filling_and_Intent_Detection_Calculate(
        path_to_label_file, path_to_predict_label_file, log_out_file)

    intent_slot_reports.show_intent_prediction_report()
    intent_slot_reports.show_slot_filling_report()

