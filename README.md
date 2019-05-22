# Template Code: BERT-for-Sequence-Labeling-and-Text-Classification
+ BERT is used for sequence annotation and text categorization template code to facilitate BERT for more tasks. Welcome to use this BERT template to solve more NLP tasks, and then share your results and code here.
+ 这是使用BERT进行序列注释和文本分类的模板代码，方便大家将BERT用于更多任务。欢迎使用这个BERT模板解决更多NLP任务，然后在这里分享你的结果和代码。
+ 项目例子具体使用方法见  [Usage example 使用方法示例.ipynb](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/blob/master/Usage%20example%20%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95%E7%A4%BA%E4%BE%8B.ipynb)

## Template Code Usage Method
1. Move google's [BERT code](https://github.com/google-research/bert) to  file ```bert``` (I've prepared a copy for you.);
2. Download google's [BERT pretrained model](https://github.com/google-research/bert) and unzip then to  file ```pretrained_model```;
3. Run Code!  You can change task_name and output_dir.

**model training**
```
python run_sequence_labeling_and_text_classification.py \
  --task_name=snips \
  --do_train=true \
  --do_eval=true \
  --data_dir=data/snips_Intent_Detection_and_Slot_Filling \
  --vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=pretrained_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --num_train_epochs=3.0 \
  --output_dir=./output_model/snips_join_task_epoch3/
```

**model prediction**
```
python run_sequence_labeling_and_text_classification.py \
  --task_name=Snips \
  --do_predict=true \
  --data_dir=data/snips_Intent_Detection_and_Slot_Filling \
  --vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output_model/snips_join_task_epoch3/model.ckpt-1000 \
  --max_seq_length=128 \
  --output_dir=./output_predict/snips_join_task_epoch3_ckpt1000
```

## File Structure

|name|function|
|-|-|
| bert |store google's [BERT code](https://github.com/google-research/bert)|||
| data |store task data set|
|output_predict|store model predict|
| output_model| store trained model|
|calculating_model_score||
|pretrained_model |store [BERT pretrained model](https://github.com/google-research/bert)|
|run_sequence_labeling.py |for Sequence Labeling Task|
|run_text_classification.py| for Text Classification Task|
|run_sequence_labeling_and_text_classification.py| for join task (come soon!)|
|calculate_model_score.py |for evaluation model |

## How to add a new task

Just write a small piece of code according to the existing template!

### Data
For example, If you have a new classification task [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs).

Before running this example you must download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e).

### Code
Now, write code!

```
class QqpProcessor(DataProcessor):
    """Processor for the QQP data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 or len(line)!=6:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            if set_type == "test":
                label = "1"
            else:
                label = tokenization.convert_to_unicode(line[5])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
 ```
 
 Registration task
 
 ```
 def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "qqp": QqpProcessor,
    }
```

### Run
```
python run_text_classification.py \
--task_name=qqp \
--do_train=true \
--do_eval=true \
--data_dir=data/snips_Intent_Detection_and_Slot_Filling \
--vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=./output/qqp_Intent_Detection/
```

## Task

Welcome to add!

|Task name|Explain|data source|
|-|-|-|
|CoNLL-2003 named entity recognition|NER||
|Atis Joint Slot Filling and Intent Prediction||https://github.com/MiuLab/SlotGated-SLU/tree/master/data/atis|
|Snips Joint Slot Filling and Intent Prediction||https://github.com/MiuLab/SlotGated-SLU/tree/master/data/snips|
|[GLUE](https://gluebenchmark.com/)|||


