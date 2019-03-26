from sklearn import metrics
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix

"""
Multiclass tf_metrics
from: 
https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py
__author__ = "Guillaume Genthial"
"""
def precision(labels, predictions, num_classes, pos_indices=None,
              weights=None, average='micro'):
    """Multi-class precision metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    pr, _, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    op, _, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (pr, op)


def recall(labels, predictions, num_classes, pos_indices=None, weights=None,
           average='micro'):
    """Multi-class recall metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, re, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    _, op, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (re, op)


def f1(labels, predictions, num_classes, pos_indices=None, weights=None,
       average='micro'):
    return fbeta(labels, predictions, num_classes, pos_indices, weights,
                 average)


def fbeta(labels, predictions, num_classes, pos_indices=None, weights=None,
          average='micro', beta=1):
    """Multi-class fbeta metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    beta : int, optional
        Weight of precision in harmonic mean
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, _, fbeta = metrics_from_confusion_matrix(
        cm, pos_indices, average=average, beta=beta)
    _, _, op = metrics_from_confusion_matrix(
        op, pos_indices, average=average, beta=beta)
    return (fbeta, op)


def safe_div(numerator, denominator):
    """Safe division, return 0 if denominator is 0"""
    numerator, denominator = tf.to_float(numerator), tf.to_float(denominator)
    zeros = tf.zeros_like(numerator, dtype=numerator.dtype)
    denominator_is_zero = tf.equal(denominator, zeros)
    return tf.where(denominator_is_zero, zeros, numerator / denominator)


def pr_re_fbeta(cm, pos_indices, beta=1):
    """Uses a confusion matrix to compute precision, recall and fbeta"""
    num_classes = cm.shape[0]
    neg_indices = [i for i in range(num_classes) if i not in pos_indices]
    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, neg_indices] = 0
    diag_sum = tf.reduce_sum(tf.diag_part(cm * cm_mask))

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[:, neg_indices] = 0
    tot_pred = tf.reduce_sum(cm * cm_mask)

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, :] = 0
    tot_gold = tf.reduce_sum(cm * cm_mask)

    pr = safe_div(diag_sum, tot_pred)
    re = safe_div(diag_sum, tot_gold)
    fbeta = safe_div((1. + beta**2) * pr * re, beta**2 * pr + re)

    return pr, re, fbeta


def metrics_from_confusion_matrix(cm, pos_indices=None, average='micro',
                                  beta=1):
    """Precision, Recall and F1 from the confusion matrix
    Parameters
    ----------
    cm : tf.Tensor of type tf.int32, of shape (num_classes, num_classes)
        The streaming confusion matrix.
    pos_indices : list of int, optional
        The indices of the positive classes
    beta : int, optional
        Weight of precision in harmonic mean
    average : str, optional
        'micro', 'macro' or 'weighted'
    """
    num_classes = cm.shape[0]
    if pos_indices is None:
        pos_indices = [i for i in range(num_classes)]

    if average == 'micro':
        return pr_re_fbeta(cm, pos_indices, beta)
    elif average in {'macro', 'weighted'}:
        precisions, recalls, fbetas, n_golds = [], [], [], []
        for idx in pos_indices:
            pr, re, fbeta = pr_re_fbeta(cm, [idx], beta)
            precisions.append(pr)
            recalls.append(re)
            fbetas.append(fbeta)
            cm_mask = np.zeros([num_classes, num_classes])
            cm_mask[idx, :] = 1
            n_golds.append(tf.to_float(tf.reduce_sum(cm * cm_mask)))

        if average == 'macro':
            pr = tf.reduce_mean(precisions)
            re = tf.reduce_mean(recalls)
            fbeta = tf.reduce_mean(fbetas)
            return pr, re, fbeta
        if average == 'weighted':
            n_gold = tf.reduce_sum(n_golds)
            pr_sum = sum(p * n for p, n in zip(precisions, n_golds))
            pr = safe_div(pr_sum, n_gold)
            re_sum = sum(r * n for r, n in zip(recalls, n_golds))
            re = safe_div(re_sum, n_gold)
            fbeta_sum = sum(f * n for f, n in zip(fbetas, n_golds))
            fbeta = safe_div(fbeta_sum, n_gold)
            return pr, re, fbeta

    else:
        raise NotImplementedError()



class Sequence_Labeling_and_Text_Classification_Calculate(object):

    def get_slot_labels(self):
        """for Sequence_Labeling labels"""
        raise NotImplementedError()

    def get_intent_labels(self):
        """for Text_Classification labels"""
        raise NotImplementedError()

    def show_intent_prediction_report(self, store_report=True):
        raise NotImplementedError()

    def show_slot_filling_report(self, store_report=True):
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

        print('混淆矩阵输出:\n', metrics.confusion_matrix(y_test_list, y_predict_list))  # 混淆矩阵输出
        print('分类报告:\n', metrics.classification_report(y_test_list, y_predict_list))  # 分类报告输出
        print("\n")

    @classmethod
    def store_model_score(cls, y_test_list=None, y_predict_list=None, label_list=None,
                          log_out_file=None, is_show_numpy_big_array=False):
        log_out_file_path = os.path.join(log_out_file, "model_score_log.txt")
        with open(log_out_file_path, "a") as log_f:
            log_f.write("时间:\t" + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + "\n")
            log_f.write('准确率:\t' + str(metrics.accuracy_score(y_test_list, y_predict_list)) + "\n")  # 预测准确率输出

            log_f.write('宏平均精确率:\t' + str(metrics.precision_score(y_test_list, y_predict_list, average='macro')) + "\n")  # 预测宏平均精确率输出
            log_f.write('微平均精确率:\t' + str(metrics.precision_score(y_test_list, y_predict_list, average='micro')) + "\n")  # 预测微平均精确率输出
            log_f.write('加权平均精确率:\t' + str(metrics.precision_score(y_test_list, y_predict_list, average='weighted')) + "\n")  # 预测加权平均精确率输出

            log_f.write('宏平均召回率:\t' + str(metrics.recall_score(y_test_list, y_predict_list, average='macro')) + "\n")  # 预测宏平均召回率输出
            log_f.write('微平均召回率:\t' + str(metrics.recall_score(y_test_list, y_predict_list, average='micro')) + "\n")  # 预测微平均召回率输出
            log_f.write('加权平均召回率:\t' + str(metrics.recall_score(y_test_list, y_predict_list, average='micro')) + "\n")  # 预测加权平均召回率输出

            log_f.write('宏平均F1-score:\t' + str(metrics.f1_score(y_test_list, y_predict_list, labels=label_list, average='macro')) + "\n")  # 预测宏平均f1-score输出
            log_f.write('微平均F1-score:\t' + str(metrics.f1_score(y_test_list, y_predict_list, labels=label_list, average='micro')) + "\n")  # 预测微平均f1-score输出
            log_f.write('加权平均F1-score:\t' + str(metrics.f1_score(y_test_list, y_predict_list, labels=label_list, average='weighted')) + "\n")  # 预测加权平均f1-score输出
            log_f.write("\n")
            log_f.write('混淆矩阵输出:\n')
            def show_numpy_big_array(a_array):
                for a_row in a_array:
                    a_row_str = [str(a_data) for a_data in a_row]
                    a_line = " ".join(a_row_str)
                    log_f.write(str(a_line) + "\n")
                log_f.write("\n")
            if is_show_numpy_big_array:
                np.set_printoptions(threshold=np.nan)
                show_numpy_big_array(metrics.confusion_matrix(y_test_list, y_predict_list))
            else:
                log_f.writelines(str(metrics.confusion_matrix(y_test_list, y_predict_list)))
            log_f.write("\n")
            log_f.write('分类报告:\n')
            classification_report = metrics.classification_report(y_test_list, y_predict_list)
            log_f.writelines(classification_report)
            log_f.write("\n\n\n")

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

    def get_conll2003_labels(self):
        return ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', "B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG",
                "I-ORG", "B-LOC", "I-LOC"]

    def show_intent_prediction_report(self, store_report=True):
        path_to_intent_label_lfile = os.path.join(self.path_to_label_file, "label")
        intent_label_list = self.get_intent_label_list(path_to_intent_label_lfile)
        path_to_predict_intent_label_file = os.path.join(self.path_to_predict_label_file, "intent_prediction_test_results.txt")
        predict_intent_label_list = self.get_predict_intent_label_list(path_to_predict_intent_label_file)
        labels = self.get_intent_labels()
        print("---show_intent_prediction_report---")
        self.show_metrics(intent_label_list, predict_intent_label_list, labels)
        print("--"*30)
        if store_report:
            self.store_model_score(intent_label_list, predict_intent_label_list, labels, self.log_out_file)

    def show_slot_filling_report(self, store_report=True, label_choose=None):
        slot_test_list, clean_predict_slot_list = self.producte_slot_list()
        slot_test_list, clean_predict_slot_list = self.delete_both_sides_is_O_word(slot_test_list, clean_predict_slot_list)
        if label_choose is None:
            labels = self.get_intent_labels()
        elif label_choose=="conll2003":
            labels = self.get_conll2003_labels()
        else:
            raise ValueError("Not found this task labels")
        print("---show_slot_filling_report---")
        self.show_metrics(slot_test_list, clean_predict_slot_list, labels)
        print("--"*30)
        if store_report:
            self.store_model_score(slot_test_list, clean_predict_slot_list, labels, self.log_out_file)

class Atis_Slot_Filling_and_Intent_Detection_Calculate(Snips_Slot_Filling_and_Intent_Detection_Calculate):

    def get_intent_labels(self):
        return ['atis_abbreviation', 'atis_aircraft', 'atis_aircraft#atis_flight#atis_flight_no',
                'atis_airfare', 'atis_airfare#atis_flight', 'atis_airfare#atis_flight_time',
                'atis_airline', 'atis_airline#atis_flight_no', 'atis_airport', 'atis_capacity',
                'atis_cheapest', 'atis_city', 'atis_day_name', 'atis_distance', 'atis_flight',
                'atis_flight#atis_airfare', 'atis_flight#atis_airline', 'atis_flight_no',
                'atis_flight_no#atis_airline', 'atis_flight_time', 'atis_ground_fare',
                'atis_ground_service', 'atis_ground_service#atis_ground_fare', 'atis_meal',
                'atis_quantity', 'atis_restriction']

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

if __name__=='__main__':
    path_to_label_file = "data/atis_Intent_Detection_and_Slot_Filling/test/"
    
    path_to_predict_label_file = "output/atis_join_task_epoch10_test1399ckpt"
    log_out_file = path_to_predict_label_file
    intent_slot_reports = Atis_Slot_Filling_and_Intent_Detection_Calculate(
        path_to_label_file, path_to_predict_label_file, log_out_file)

    intent_slot_reports.show_intent_prediction_report(store_report=True)
    intent_slot_reports.show_slot_filling_report(store_report=True)
    #print(intent_slot_reports.get_intent_labels())
    #print(intent_slot_reports.get_slot_labels())
