#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:yuanxiao
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
import tf_metrics
import pickle
import shutil

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", None, "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-4, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 5.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores", 8,
                     "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """read raw data set."""
        raise NotImplementedError()

class Atis_Slot_Filling_Processor(DataProcessor):

    def get_examples(self, data_dir):
        with open(os.path.join(data_dir, "seq.in")) as seq_in_f:
            with open(os.path.join(data_dir, "seq.out")) as seq_out_f:
                with open(os.path.join(data_dir, "label")) as label_f:
                    seq_in_list = [seq.replace("\n", '') for seq in seq_in_f.readlines()]
                    seq_out_list = [seq.replace("\n", '') for seq in seq_out_f.readlines()]
                    seq_label_list = [seq.replace("\n", '') for seq in label_f.readlines()]
                    assert len(seq_in_list) == len(seq_out_list)
                    assert len(seq_in_list) == len(seq_label_list)
                    lines = list(zip(seq_in_list, seq_out_list, seq_label_list))
                    return lines

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "test")), "test")

    def get_labels_from_files(self, data_dir):
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

    def get_labels(self):
        return ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', 'B-aircraft_code', 'B-airline_code', 'B-airline_name', 'B-airport_code', 'B-airport_name', 'B-arrive_date.date_relative', 'B-arrive_date.day_name', 'B-arrive_date.day_number', 'B-arrive_date.month_name', 'B-arrive_date.today_relative', 'B-arrive_time.end_time', 'B-arrive_time.period_mod', 'B-arrive_time.period_of_day', 'B-arrive_time.start_time', 'B-arrive_time.time', 'B-arrive_time.time_relative', 'B-booking_class', 'B-city_name', 'B-class_type', 'B-compartment', 'B-connect', 'B-cost_relative', 'B-day_name', 'B-day_number', 'B-days_code', 'B-depart_date.date_relative', 'B-depart_date.day_name', 'B-depart_date.day_number', 'B-depart_date.month_name', 'B-depart_date.today_relative', 'B-depart_date.year', 'B-depart_time.end_time', 'B-depart_time.period_mod', 'B-depart_time.period_of_day', 'B-depart_time.start_time', 'B-depart_time.time', 'B-depart_time.time_relative', 'B-economy', 'B-fare_amount', 'B-fare_basis_code', 'B-flight', 'B-flight_days', 'B-flight_mod', 'B-flight_number', 'B-flight_stop', 'B-flight_time', 'B-fromloc.airport_code', 'B-fromloc.airport_name', 'B-fromloc.city_name', 'B-fromloc.state_code', 'B-fromloc.state_name', 'B-meal', 'B-meal_code', 'B-meal_description', 'B-mod', 'B-month_name', 'B-or', 'B-period_of_day', 'B-restriction_code', 'B-return_date.date_relative', 'B-return_date.day_name', 'B-return_date.day_number', 'B-return_date.month_name', 'B-return_date.today_relative', 'B-return_time.period_mod', 'B-return_time.period_of_day', 'B-round_trip', 'B-state_code', 'B-state_name', 'B-stoploc.airport_code', 'B-stoploc.airport_name', 'B-stoploc.city_name', 'B-stoploc.state_code', 'B-time', 'B-time_relative', 'B-today_relative', 'B-toloc.airport_code', 'B-toloc.airport_name', 'B-toloc.city_name', 'B-toloc.country_name', 'B-toloc.state_code', 'B-toloc.state_name', 'B-transport_type', 'I-airline_name', 'I-airport_name', 'I-arrive_date.day_number', 'I-arrive_time.end_time', 'I-arrive_time.period_of_day', 'I-arrive_time.start_time', 'I-arrive_time.time', 'I-arrive_time.time_relative', 'I-city_name', 'I-class_type', 'I-cost_relative', 'I-depart_date.day_number', 'I-depart_date.today_relative', 'I-depart_time.end_time', 'I-depart_time.period_of_day', 'I-depart_time.start_time', 'I-depart_time.time', 'I-depart_time.time_relative', 'I-economy', 'I-fare_amount', 'I-fare_basis_code', 'I-flight_mod', 'I-flight_number', 'I-flight_stop', 'I-flight_time', 'I-fromloc.airport_name', 'I-fromloc.city_name', 'I-fromloc.state_name', 'I-meal_code', 'I-meal_description', 'I-restriction_code', 'I-return_date.date_relative', 'I-return_date.day_number', 'I-return_date.today_relative', 'I-round_trip', 'I-state_name', 'I-stoploc.city_name', 'I-time', 'I-today_relative', 'I-toloc.airport_name', 'I-toloc.city_name', 'I-toloc.state_name', 'I-transport_type', 'O']

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

class CoNLL2003NERProcessor(DataProcessor):

    def get_examples(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                if len(contends) == 0 and words[-1] == '.':
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines

    def get_train_examples(self, data_dir):
        return self._create_example(
            self.get_examples(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self.get_examples(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self,data_dir):
        return self._create_example(
            self.get_examples(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', "B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

class Snips_Slot_Filling_Processor(DataProcessor):

    def get_examples(self, data_dir):
        path_seq_in = os.path.join(data_dir, "seq.in")
        path_seq_out = os.path.join(data_dir, "seq.out")
        seq_in_list, seq_out_list = [], []
        with open(path_seq_in) as seq_in_f:
            with open(path_seq_out) as seq_out_f:
                for seqin, seqout in zip(seq_in_f.readlines(), seq_out_f.readlines()):
                    seqin_words = [word for word in seqin.split() if len(word) > 0]
                    seqout_words = [word for word in seqout.split() if len(word) > 0]
                    assert len(seqin_words) == len(seqout_words)
                    seq_in_list.append(" ".join(seqin_words))
                    seq_out_list.append(" ".join(seqout_words))
        lines = list(zip(seq_in_list, seq_out_list))
        return lines

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "test")), "test")

    def get_labels_from_files(self, data_dir):
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

    def get_labels(self):
        return ['[Padding]', '[##WordPiece]', '[CLS]', '[SEP]', 'B-album', 'B-artist', 'B-best_rating', 'B-city', 'B-condition_description', 'B-condition_temperature', 'B-country', 'B-cuisine', 'B-current_location', 'B-entity_name', 'B-facility', 'B-genre', 'B-geographic_poi', 'B-location_name', 'B-movie_name', 'B-movie_type', 'B-music_item', 'B-object_location_type', 'B-object_name', 'B-object_part_of_series_type', 'B-object_select', 'B-object_type', 'B-party_size_description', 'B-party_size_number', 'B-playlist', 'B-playlist_owner', 'B-poi', 'B-rating_unit', 'B-rating_value', 'B-restaurant_name', 'B-restaurant_type', 'B-served_dish', 'B-service', 'B-sort', 'B-spatial_relation', 'B-state', 'B-timeRange', 'B-track', 'B-year', 'I-album', 'I-artist', 'I-city', 'I-country', 'I-cuisine', 'I-current_location', 'I-entity_name', 'I-facility', 'I-genre', 'I-geographic_poi', 'I-location_name', 'I-movie_name', 'I-movie_type', 'I-music_item', 'I-object_location_type', 'I-object_name', 'I-object_part_of_series_type', 'I-object_select', 'I-object_type', 'I-party_size_description', 'I-playlist', 'I-playlist_owner', 'I-poi', 'I-restaurant_name', 'I-restaurant_type', 'I-served_dish', 'I-service', 'I-sort', 'I-spatial_relation', 'I-state', 'I-timeRange', 'I-track', 'O']

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(os.path.join(FLAGS.output_dir, "label2id.pkl"), 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("[##WordPiece]")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])

    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("[Padding]")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    write_tokens(ntokens, mode)
    return feature


def write_tokens(tokens, mode):
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
        wf = open(path, 'a')
        for token in tokens:
            if token != "[Padding]":
                wf.write(token + '\n')
        wf.close()


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predicts = tf.argmax(probabilities, axis=-1)
        return (loss, per_example_loss, logits, predicts)
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
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

            def metric_fn(per_example_loss, label_ids, logits):
                # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                pos_indices_list = list(range(num_labels))[4:]  #["[Padding]","[##WordPiece]", "[CLS]", "[SEP]"] + seq_out_set
                pos_indices_list = pos_indices_list[:-1]    # do not care "O"
                precision_macro = tf_metrics.precision(label_ids, predictions, num_labels, pos_indices_list, average="macro")
                recall_macro = tf_metrics.recall(label_ids, predictions, num_labels, pos_indices_list, average="macro")
                f_macro = tf_metrics.f1(label_ids, predictions, num_labels, pos_indices_list, average="macro")
                precision_micro = tf_metrics.precision(label_ids, predictions, num_labels, pos_indices_list, average="micro")
                recall_micro = tf_metrics.recall(label_ids, predictions, num_labels, pos_indices_list, average="micro")
                f_micro = tf_metrics.f1(label_ids, predictions, num_labels, pos_indices_list, average="micro")
                return {
                    "eval_precision(macro)": precision_macro,
                    "eval_recall(macro)": recall_macro,
                    "eval_f(macro)": f_macro,
                    "eval_precision(micro)": precision_micro,
                    "eval_recall(micro)": recall_micro,
                    "eval_f(micro)": f_micro,
                    "eval_loss": per_example_loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def main(_):
    # ----------------for test------------------
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
    # ----------------for test------------------

    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "atis": Atis_Slot_Filling_Processor,
        "snips": Snips_Slot_Filling_Processor,
        "conll2003ner": CoNLL2003NERProcessor,
    }
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    id2label = {}
    for (i, label) in enumerate(label_list):
        id2label[i] = label

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
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
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
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "slot_filling_test_results.txt")
        with open(output_predict_file, 'w') as writer:
            for prediction in result:
                output_line = " ".join(id2label[id] for id in prediction if id != 0) + "\n"  #0--->"[Padding]"
                writer.write(output_line)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()


