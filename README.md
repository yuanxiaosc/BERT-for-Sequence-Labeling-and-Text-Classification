# BERT-for-Sequence-Labeling-and-Text-Classification
+ BERT is used for sequence annotation and text categorization template code to facilitate BERT for more tasks. Welcome to use this BERT template to solve more NLP tasks, and then share your results and code here.
+ 这是使用BERT进行序列注释和文本分类的模板代码，以使用BERT执行更多任务。欢迎使用这个BERT模板解决更多NLP任务，然后在这里分享你的结果和代码。

## Template Code Usage Method
1. Move google's [BERT code](https://github.com/google-research/bert) to  file ```bert``` (I've prepared a copy for you.);
2. Download google's [BERT pretrained model](https://github.com/google-research/bert) and unzip then to  file ```pretrained_model```;
3. Run Code!

   ```
  python run_text_classification.py \
  --task_name=Snips \
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
  --output_dir=./output/snips_Intent_Detection/
  ```
  
## File Structure

   BERT-for-Sequence-Labeling-and-Text-Classification
  |____ bert store google's [BERT code](https://github.com/google-research/bert)
  |____ data store task data set
  |____ output store model output
  |____ pretrained_model store [BERT pretrained model](https://github.com/google-research/bert)
  |____ run_sequence_labeling.py for Sequence Labeling Task
  |____ run_text_classification.py for Text Classification Task
  |____ run_sequence_labeling_and_text_classification.py for join task (come soon!)  
  |____ tf_metrics.py for evaluation model 
    
## Task

Welcome to add!

|Task name|Explain|data source|
|-|-|-|
|CoNLL-2003 named entity recognition|NER||
|Atis Joint Slot Filling and Intent Prediction||https://github.com/MiuLab/SlotGated-SLU/tree/master/data/atis|
|Snips Joint Slot Filling and Intent Prediction||https://github.com/MiuLab/SlotGated-SLU/tree/master/data/snips|
|[GLUE](https://gluebenchmark.com/)|||
