#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:yuanxiao
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
import pickle
import shutil
import calculate_model_score as tf_metrics

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_bool(
    "calculate_model_score", True, "calculate_model_score_on_test_data")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the input sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the labeling sequence(Slot Filling).
            specified for train and dev examples, but not for test examples.
          label: (Optional) string. The label(Intent Prediction) of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 slot_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.slot_ids = slot_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class Atis_Slot_Filling_and_Intent_Detection_Processor(DataProcessor):

    def get_examples(self, data_dir):
        path_seq_in = os.path.join(data_dir, "seq.in")
        path_seq_out = os.path.join(data_dir, "seq.out")
        path_label = os.path.join(data_dir, "label")
        seq_in_list, seq_out_list, label_list = [], [], []
        with open(path_seq_in) as seq_in_f:
            with open(path_seq_out) as seq_out_f:
                with open(path_label) as label_f:
                    for seqin, seqout, label in zip(seq_in_f.readlines(), seq_out_f.readlines(), label_f.readlines()):
                        seqin_words = [word for word in seqin.split() if len(word) > 0]
                        seqout_words = [word for word in seqout.split() if len(word) > 0]
                        label_list.append(label.replace("\n", ""))
                        assert len(seqin_words) == len(seqout_words)
                        seq_in_list.append(" ".join(seqin_words))
                        seq_out_list.append(" ".join(seqout_words))
            lines = list(zip(seq_in_list, seq_out_list, label_list))
            return lines

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "test")), "test")

    def get_slot_labels_from_files(self, data_dir):
        label_set = set()
        for f_type in ["train", "valid", "test"]:
            seq_out_dir = os.path.join(os.path.join(data_dir, f_type), "seq.out")
            with open(seq_out_dir) as data_f:
                seq_sentence_list = [seq.split() for seq in data_f.readlines()]
                seq_word_list = [word for seq in seq_sentence_list for word in seq]
                label_set = label_set | set(seq_word_list)
        label_list = list(label_set)
        label_list.sort()
        return ["[Padding]", "[##WordPiece]", "[CLS]", "[SEP]"] + label_list

    def get_slot_labels(self):
        return ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', 'B-aircraft_code', 'B-airline_code', 'B-airline_name',
                'B-airport_code', 'B-airport_name', 'B-arrive_date.date_relative', 'B-arrive_date.day_name',
                'B-arrive_date.day_number', 'B-arrive_date.month_name', 'B-arrive_date.today_relative',
                'B-arrive_time.end_time', 'B-arrive_time.period_mod', 'B-arrive_time.period_of_day',
                'B-arrive_time.start_time', 'B-arrive_time.time', 'B-arrive_time.time_relative', 'B-booking_class',
                'B-city_name', 'B-class_type', 'B-compartment', 'B-connect', 'B-cost_relative', 'B-day_name',
                'B-day_number', 'B-days_code', 'B-depart_date.date_relative', 'B-depart_date.day_name',
                'B-depart_date.day_number', 'B-depart_date.month_name', 'B-depart_date.today_relative',
                'B-depart_date.year', 'B-depart_time.end_time', 'B-depart_time.period_mod',
                'B-depart_time.period_of_day', 'B-depart_time.start_time', 'B-depart_time.time',
                'B-depart_time.time_relative', 'B-economy', 'B-fare_amount', 'B-fare_basis_code', 'B-flight',
                'B-flight_days', 'B-flight_mod', 'B-flight_number', 'B-flight_stop', 'B-flight_time',
                'B-fromloc.airport_code', 'B-fromloc.airport_name', 'B-fromloc.city_name', 'B-fromloc.state_code',
                'B-fromloc.state_name', 'B-meal', 'B-meal_code', 'B-meal_description', 'B-mod', 'B-month_name', 'B-or',
                'B-period_of_day', 'B-restriction_code', 'B-return_date.date_relative', 'B-return_date.day_name',
                'B-return_date.day_number', 'B-return_date.month_name', 'B-return_date.today_relative',
                'B-return_time.period_mod', 'B-return_time.period_of_day', 'B-round_trip', 'B-state_code',
                'B-state_name', 'B-stoploc.airport_code', 'B-stoploc.airport_name', 'B-stoploc.city_name',
                'B-stoploc.state_code', 'B-time', 'B-time_relative', 'B-today_relative', 'B-toloc.airport_code',
                'B-toloc.airport_name', 'B-toloc.city_name', 'B-toloc.country_name', 'B-toloc.state_code',
                'B-toloc.state_name', 'B-transport_type', 'I-airline_name', 'I-airport_name',
                'I-arrive_date.day_number', 'I-arrive_time.end_time', 'I-arrive_time.period_of_day',
                'I-arrive_time.start_time', 'I-arrive_time.time', 'I-arrive_time.time_relative', 'I-city_name',
                'I-class_type', 'I-cost_relative', 'I-depart_date.day_number', 'I-depart_date.today_relative',
                'I-depart_time.end_time', 'I-depart_time.period_of_day', 'I-depart_time.start_time',
                'I-depart_time.time', 'I-depart_time.time_relative', 'I-economy', 'I-fare_amount', 'I-fare_basis_code',
                'I-flight_mod', 'I-flight_number', 'I-flight_stop', 'I-flight_time', 'I-fromloc.airport_name',
                'I-fromloc.city_name', 'I-fromloc.state_name', 'I-meal_code', 'I-meal_description',
                'I-restriction_code', 'I-return_date.date_relative', 'I-return_date.day_number',
                'I-return_date.today_relative', 'I-round_trip', 'I-state_name', 'I-stoploc.city_name', 'I-time',
                'I-today_relative', 'I-toloc.airport_name', 'I-toloc.city_name', 'I-toloc.state_name',
                'I-transport_type', 'O']

    def get_intent_labels(self):
        return ['atis_abbreviation', 'atis_aircraft', 'atis_aircraft#atis_flight#atis_flight_no',
                'atis_airfare', 'atis_airfare#atis_flight', 'atis_airfare#atis_flight_time',
                'atis_airline', 'atis_airline#atis_flight_no', 'atis_airport', 'atis_capacity',
                'atis_cheapest', 'atis_city', 'atis_day_name', 'atis_distance', 'atis_flight',
                'atis_flight#atis_airfare', 'atis_flight#atis_airline', 'atis_flight_no',
                'atis_flight_no#atis_airline', 'atis_flight_time', 'atis_ground_fare',
                'atis_ground_service', 'atis_ground_service#atis_ground_fare', 'atis_meal',
                'atis_quantity', 'atis_restriction']

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class Snips_Slot_Filling_and_Intent_Detection_Processor(DataProcessor):

    def get_examples(self, data_dir):
        path_seq_in = os.path.join(data_dir, "seq.in")
        path_seq_out = os.path.join(data_dir, "seq.out")
        path_label = os.path.join(data_dir, "label")
        seq_in_list, seq_out_list, label_list = [], [], []
        with open(path_seq_in) as seq_in_f:
            with open(path_seq_out) as seq_out_f:
                with open(path_label) as label_f:
                    for seqin, seqout, label in zip(seq_in_f.readlines(), seq_out_f.readlines(), label_f.readlines()):
                        seqin_words = [word for word in seqin.split() if len(word) > 0]
                        seqout_words = [word for word in seqout.split() if len(word) > 0]
                        label_list.append(label.replace("\n", ""))
                        assert len(seqin_words) == len(seqout_words)
                        seq_in_list.append(" ".join(seqin_words))
                        seq_out_list.append(" ".join(seqout_words))
            lines = list(zip(seq_in_list, seq_out_list, label_list))
            return lines

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "test")), "test")

    def get_slot_labels_from_files(self, data_dir):
        label_set = set()
        for f_type in ["train", "valid", "test"]:
            seq_out_dir = os.path.join(os.path.join(data_dir, f_type), "seq.out")
            with open(seq_out_dir) as data_f:
                seq_sentence_list = [seq.split() for seq in data_f.readlines()]
                seq_word_list = [word for seq in seq_sentence_list for word in seq]
                label_set = label_set | set(seq_word_list)
        label_list = list(label_set)
        label_list.sort()
        return ["[Padding]", "[##WordPiece]", "[CLS]", "[SEP]"] + label_list

    def get_slot_labels(self):
        return ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', 'B-album', 'B-artist', 'B-best_rating', 'B-city', 'B-condition_description', 'B-condition_temperature', 'B-country', 'B-cuisine', 'B-current_location', 'B-entity_name', 'B-facility', 'B-genre', 'B-geographic_poi', 'B-location_name', 'B-movie_name', 'B-movie_type', 'B-music_item', 'B-object_location_type', 'B-object_name', 'B-object_part_of_series_type', 'B-object_select', 'B-object_type', 'B-party_size_description', 'B-party_size_number', 'B-playlist', 'B-playlist_owner', 'B-poi', 'B-rating_unit', 'B-rating_value', 'B-restaurant_name', 'B-restaurant_type', 'B-served_dish', 'B-service', 'B-sort', 'B-spatial_relation', 'B-state', 'B-timeRange', 'B-track', 'B-year', 'I-album', 'I-artist', 'I-city', 'I-country', 'I-cuisine', 'I-current_location', 'I-entity_name', 'I-facility', 'I-genre', 'I-geographic_poi', 'I-location_name', 'I-movie_name', 'I-movie_type', 'I-music_item', 'I-object_location_type', 'I-object_name', 'I-object_part_of_series_type', 'I-object_select', 'I-object_type', 'I-party_size_description', 'I-playlist', 'I-playlist_owner', 'I-poi', 'I-restaurant_name', 'I-restaurant_type', 'I-served_dish', 'I-service', 'I-sort', 'I-spatial_relation', 'I-state', 'I-timeRange', 'I-track', 'O']

    def get_intent_labels(self):
        return ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_single_example(ex_index, example, slot_label_list, intent_label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            slot_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    slot_label_map = {}
    for (i, label) in enumerate(slot_label_list):
        slot_label_map[label] = i
    with open(os.path.join(FLAGS.output_dir, "slot_label2id.pkl"), 'wb') as w:
        pickle.dump(slot_label_map, w)

    intent_label_map = {}
    for (i, label) in enumerate(intent_label_list):
        intent_label_map[label] = i
    with open(os.path.join(FLAGS.output_dir, "intent_label2id.pkl"), 'wb') as w:
        pickle.dump(intent_label_map, w)

    text_a_list = example.text_a.split(" ")
    text_b_list = example.text_b.split(" ")

    tokens_a = []
    slots_b = []
    for i, word in enumerate(text_a_list):
        token_a = tokenizer.tokenize(word)
        tokens_a.extend(token_a)
        slot_i = text_b_list[i]
        for m in range(len(token_a)):
            if m == 0:
                slots_b.append(slot_i)
            else:
                slots_b.append("[##WordPiece]")
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
        slots_b = slots_b[0:(max_seq_length - 2)]

    tokens = []
    slot_ids = []
    segment_ids = []
    tokens.append("[CLS]")
    slot_ids.append(slot_label_map["[CLS]"])
    segment_ids.append(0)
    for i, token in enumerate(tokens_a):
        tokens.append(token)
        segment_ids.append(0)
        slot_ids.append(slot_label_map[slots_b[i]])

    tokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    slot_ids.append(slot_label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        slot_ids.append(0)
        tokens.append("[Padding]")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(slot_ids) == max_seq_length

    label_id = intent_label_map[example.label]

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("slots_ids: %s" % " ".join([str(x) for x in slot_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        slot_ids=slot_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)

    return feature

def file_based_convert_examples_to_features(
        examples, slot_label_list, intent_label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, slot_label_list, intent_label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["slot_ids"] = create_int_feature(feature.slot_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "slot_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 slot_label_ids, intent_label_ids, num_slot_labels, num_intent_labels,
                 use_one_hot_embeddings):
    """Creates a sequence labeling and classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token. float Tensor of shape [batch_size, hidden_size]
    intent_output_layer = model.get_pooled_output()

    intent_hidden_size = intent_output_layer.shape[-1].value

    intent_output_weights = tf.get_variable(
        "intent_output_weights", [num_intent_labels, intent_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    intent_output_bias = tf.get_variable(
        "intent_output_bias", [num_intent_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("intent_loss"):
        if is_training:
            # I.e., 0.1 dropout
            intent_output_layer = tf.nn.dropout(intent_output_layer, keep_prob=0.9)

        intent_logits = tf.matmul(intent_output_layer, intent_output_weights, transpose_b=True)
        intent_logits = tf.nn.bias_add(intent_logits, intent_output_bias)
        intent_probabilities = tf.nn.softmax(intent_logits, axis=-1)
        intent_log_probs = tf.nn.log_softmax(intent_logits, axis=-1)
        intent_predictions = tf.argmax(intent_logits, axis=-1)
        intent_one_hot_labels = tf.one_hot(intent_label_ids, depth=num_intent_labels, dtype=tf.float32)
        intent_per_example_loss = -tf.reduce_sum(intent_one_hot_labels * intent_log_probs, axis=-1)
        intent_loss = tf.reduce_mean(intent_per_example_loss)
        #return (intent_loss, intent_per_example_loss, intent_logits, intent_probabilities, intent_predictions)

    #     """Gets final hidden layer of encoder.
    #
    #     Returns:
    #       float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
    #       to the final hidden of the transformer encoder.
    #     """
    slot_output_layer = model.get_sequence_output()

    ###################################################################################
    with tf.variable_scope("slot_output_layers"):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=384)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=384)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw, inputs=slot_output_layer, dtype=tf.float32)
        slot_output_layer = tf.concat([outputs[0], outputs[1]], axis=2)
    ###################################################################################

    slot_hidden_size = slot_output_layer.shape[-1].value

    slot_output_weight = tf.get_variable(
        "slot_output_weights", [num_slot_labels, slot_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    slot_output_bias = tf.get_variable(
        "slot_output_bias", [num_slot_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("slot_loss"):
        if is_training:
            slot_output_layer = tf.nn.dropout(slot_output_layer, keep_prob=0.9)
        slot_output_layer = tf.reshape(slot_output_layer, [-1, slot_hidden_size])
        slot_logits = tf.matmul(slot_output_layer, slot_output_weight, transpose_b=True)
        slot_logits = tf.nn.bias_add(slot_logits, slot_output_bias)
        slot_logits = tf.reshape(slot_logits, [-1, FLAGS.max_seq_length, num_slot_labels])
        slot_log_probs = tf.nn.log_softmax(slot_logits, axis=-1)
        slot_one_hot_labels = tf.one_hot(slot_label_ids, depth=num_slot_labels, dtype=tf.float32)
        slot_per_example_loss = -tf.reduce_sum(slot_one_hot_labels * slot_log_probs, axis=-1)
        slot_loss = tf.reduce_sum(slot_per_example_loss)
        slot_probabilities = tf.nn.softmax(slot_logits, axis=-1)
        slot_predictions = tf.argmax(slot_probabilities, axis=-1)
        #return (slot_loss, slot_per_example_loss, slot_logits, slot_predict)

    loss = intent_loss + slot_loss
    return (loss,
            intent_loss, intent_per_example_loss, intent_logits, intent_predictions,
            slot_loss, slot_per_example_loss, slot_logits, slot_predictions)


def model_fn_builder(bert_config, num_slot_labels, num_intent_labels, init_checkpoint,
                     learning_rate, num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        slot_label_ids = features["slot_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        intent_label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(intent_label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss,
         intent_loss, intent_per_example_loss, intent_logits, intent_predictions,
         slot_loss, slot_per_example_loss, slot_logits, slot_predictions) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            slot_label_ids, intent_label_ids, num_slot_labels, num_intent_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(intent_per_example_loss, intent_label_ids, intent_logits,
                          slot_per_example_loss, slot_label_ids, slot_logits, is_real_example):
                intent_predictions = tf.argmax(intent_logits, axis=-1, output_type=tf.int32)
                intent_accuracy = tf.metrics.accuracy(
                    labels=intent_label_ids, predictions=intent_predictions, weights=is_real_example)
                intent_loss = tf.metrics.mean(values=intent_per_example_loss, weights=is_real_example)
                slot_predictions = tf.argmax(slot_logits, axis=-1, output_type=tf.int32)
                slot_pos_indices_list = list(range(num_slot_labels))[4:]  # ["[Padding]","[##WordPiece]", "[CLS]", "[SEP]"] + seq_out_set
                pos_indices_list = slot_pos_indices_list[:-1]  # do not care "O"
                slot_precision_macro = tf_metrics.precision(slot_label_ids, slot_predictions, num_slot_labels,
                                                      slot_pos_indices_list, average="macro")
                slot_recall_macro = tf_metrics.recall(slot_label_ids, slot_predictions, num_slot_labels,
                                                slot_pos_indices_list, average="macro")
                slot_f_macro = tf_metrics.f1(slot_label_ids, slot_predictions, num_slot_labels, slot_pos_indices_list,
                                       average="macro")
                slot_precision_micro = tf_metrics.precision(slot_label_ids, slot_predictions, num_slot_labels,
                                                      slot_pos_indices_list, average="micro")
                slot_recall_micro = tf_metrics.recall(slot_label_ids, slot_predictions, num_slot_labels,
                                                slot_pos_indices_list, average="micro")
                slot_f_micro = tf_metrics.f1(slot_label_ids, slot_predictions, num_slot_labels, slot_pos_indices_list,
                                       average="micro")
                slot_loss = tf.metrics.mean(values=slot_per_example_loss, weights=is_real_example)
                return {
                    "eval_intent_accuracy": intent_accuracy,
                    "eval_intent_loss": intent_loss,
                    "eval_slot_precision(macro)": slot_precision_macro,
                    "eval_slot_recall(macro)": slot_recall_macro,
                    "eval_slot_f(macro)": slot_f_macro,
                    "eval_slot_precision(micro)": slot_precision_micro,
                    "eval_slot_recall(micro)": slot_recall_micro,
                    "eval_slot_f(micro)": slot_f_micro,
                    "eval_slot_loss": slot_loss,
                }

            eval_metrics = (metric_fn,
                            [intent_per_example_loss, intent_label_ids, intent_logits,
                             slot_per_example_loss, slot_label_ids, slot_logits, is_real_example])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"intent_predictions": intent_predictions,
                             "slot_predictions": slot_predictions},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn

def main(_):
    # ----------------for code test------------------
    if FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):
            try:
                os.removedirs(FLAGS.output_dir)
                os.makedirs(FLAGS.output_dir)
            except:
                tf.logging.info("***** Running evaluation *****")
                tf.logging.warning(FLAGS.output_dir + " is  not empty, here use shutil.rmtree(FLAGS.output_dir)!")
                shutil.rmtree(FLAGS.output_dir)
                os.makedirs(FLAGS.output_dir)
        else:
            os.makedirs(FLAGS.output_dir)
    # ----------------for code test------------------


    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "atis": Atis_Slot_Filling_and_Intent_Detection_Processor,
        "snips": Snips_Slot_Filling_and_Intent_Detection_Processor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    intent_label_list = processor.get_intent_labels()
    slot_label_list = processor.get_slot_labels()

    intent_id2label = {}
    for (i, label) in enumerate(intent_label_list):
        intent_id2label[i] = label
    slot_id2label = {}
    for (i, label) in enumerate(slot_label_list):
        slot_id2label[i] = label

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_slot_labels=len(slot_label_list),
        num_intent_labels=len(intent_label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, slot_label_list, intent_label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, slot_label_list, intent_label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, slot_label_list, intent_label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        #result_list = list(result)
        #with open(os.path.join(FLAGS.output_dir, "all_test_results.pkl"), "wb") as result_f:
        #    pickle.dump(result_list, result_f)

        intent_output_predict_file = os.path.join(FLAGS.output_dir, "intent_prediction_test_results.txt")
        slot_output_predict_file = os.path.join(FLAGS.output_dir, "slot_filling_test_results.txt")
        with tf.gfile.GFile(intent_output_predict_file, "w") as intent_writer:
            with tf.gfile.GFile(slot_output_predict_file, "w") as slot_writer:
                num_written_lines = 0
                tf.logging.info("***** Intent Predict and Slot Filling results *****")
                for (i, prediction) in enumerate(result):
                    intent_prediction = prediction["intent_predictions"]
                    slot_predictions = prediction["slot_predictions"]
                    if i >= num_actual_predict_examples:
                        break

                    intent_output_line = str(intent_id2label[intent_prediction]) + "\n"
                    intent_writer.write(intent_output_line)

                    slot_output_line = " ".join(
                        slot_id2label[id] for id in slot_predictions if id != 0) + "\n"  # 0--->"[Padding]"
                    slot_writer.write(slot_output_line)

                    num_written_lines += 1
                assert num_written_lines == num_actual_predict_examples

        if FLAGS.calculate_model_score:
            path_to_label_file = os.path.join(FLAGS.data_dir, "test")
            path_to_predict_label_file = FLAGS.output_dir
            log_out_file = path_to_predict_label_file
            if FLAGS.task_name.lower() == "snips":
                intent_slot_reports = tf_metrics.Snips_Slot_Filling_and_Intent_Detection_Calculate(
                    path_to_label_file, path_to_predict_label_file, log_out_file)
            elif FLAGS.task_name.lower() == "atis":
                intent_slot_reports = tf_metrics.Atis_Slot_Filling_and_Intent_Detection_Calculate(
                    path_to_label_file, path_to_predict_label_file, log_out_file)
            else:
                raise ValueError("Not this calculate_model_score")
            intent_slot_reports.show_intent_prediction_report(store_report=True)
            intent_slot_reports.show_slot_filling_report(store_report=True)



if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
