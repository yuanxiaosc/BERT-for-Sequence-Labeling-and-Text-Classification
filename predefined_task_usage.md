## BERT-for-Sequence-Labeling-and-Text-Classification

## BERT information

Take uncased_L-12_H-768_A-12 as an example, which contains the following three files:
+ uncased_L-12_H-768_A-12/vocab.txt
+ uncased_L-12_H-768_A-12/bert_config.json
+ uncased_L-12_H-768_A-12/bert_model.ckpt

## Sequence-Labeling-task 序列标注任务

> Examples of model training usage

### ATIS
python run_sequence_labeling.py \
--task_name="atis" \
--do_train=True \
--do_eval=True \
--do_predict=True \
--data_dir=data/atis_Intent_Detection_and_Slot_Filling \
--vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=./output_model/atis_Slot_Filling_epoch3/ 
### SNIPS
python run_sequence_labeling.py \
--task_name="snips" \
--do_train=True \
--do_eval=True \
--do_predict=True \
--data_dir=data/snips_Intent_Detection_and_Slot_Filling \
--vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=./output_model/snips_Slot_Filling_epochs3/ 
### CoNLL2003NER
python run_sequence_labeling.py \
--task_name="conll2003ner" \
--do_train=True \
--do_eval=True \
--do_predict=True \
--data_dir=data/CoNLL2003_NER \
--vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=./output_model/conll2003ner_epoch3/ 
## Sequence labeling task prediction 序列标注任务预测 
python run_sequence_labeling.py \
--task_name="conll2003ner" \
--do_predict=True \
--data_dir=data/CoNLL2003_NER \
--vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=output_model/conll2003ner_epoch3/model.ckpt-653 \
--output_dir=./output_predict/conll2003ner_epoch3_ckpt653/ 
## Text-Classification Train 文本分类任务训练 

### ATIS Train
python run_text_classification.py \
  --task_name=atis \
  --do_train=true \
  --do_eval=true \
  --data_dir=data/atis_Intent_Detection_and_Slot_Filling \
  --vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=pretrained_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./output_model/atis_Intent_Detection_epochs3/
### ATIS Make Predicte
python run_text_classification.py \
  --task_name=atis \
  --do_predict=true \
  --data_dir=data/atis_Intent_Detection_and_Slot_Filling \
  --vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output_model/atis_Intent_Detection_epochs3/model.ckpt-419 \
  --max_seq_length=128 \
  --output_dir=./output_predict/atis_Intent_Detection_epoch3_ckpt419
### SNIPS Make Predicte
python run_text_classification.py \
  --task_name=Snips \
  --do_predict=true \
  --data_dir=data/snips_Intent_Detection_and_Slot_Filling \
  --vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output_model/snips_Intent_Detection_epochs3/model.ckpt-1226 \
  --max_seq_length=128 \
  --output_dir=./output_predict/snips_Intent_Detection_epoch3_ckpt1226/
## Joint task training 联合任务训练

### SNIPS Train
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
### ATIS Train
python run_sequence_labeling_and_text_classification.py \
  --task_name=Atis \
  --do_train=true \
  --do_eval=true \
  --data_dir=data/atis_Intent_Detection_and_Slot_Filling \
  --vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=pretrained_model/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --num_train_epochs=3.0 \
  --output_dir=./output_model/atis_join_task_epoch3/
### ATIS Next Train
python run_sequence_labeling_and_text_classification.py \
  --task_name=Atis \
  --do_train=true \
  --do_eval=true \
  --data_dir=data/atis_Intent_Detection_and_Slot_Filling \
  --vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output_model/atis_join_task_epoch3/model.ckpt-1399 \
  --num_train_epochs=3.0 \
  --output_dir=./output_model/atis_join_task_epoch6/
## Joint Mission predict 联合任务预测 

### SNIPS Make Predicte
python run_sequence_labeling_and_text_classification.py \
  --task_name=Snips \
  --do_predict=true \
  --data_dir=data/snips_Intent_Detection_and_Slot_Filling \
  --vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output_model/snips_join_task_epoch3/model.ckpt-1000 \
  --max_seq_length=128 \
  --output_dir=./output_predict/snips_join_task_epoch3_ckpt1000
### ATIS Make Predicte
python run_sequence_labeling_and_text_classification.py \
  --task_name=Atis \
  --do_predict=true \
  --data_dir=data/atis_Intent_Detection_and_Slot_Filling \
  --vocab_file=pretrained_model/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output_model/atis_join_task_epoch3/model.ckpt-1000 \
  --max_seq_length=128 \
  --output_dir=./output_predict/atis_join_task_epoch30_ckpt1000