# Multimodal Semantic Membership Inference Attack (MSMIA)
This repository contains implementation for work Multimodal Semantic Membership Inference Attacks (MSMIAs)
## Description
The system is the first collection of models and pipelines for membership inference attacks against large language models, 
built initially with a priority for the Russian language, and extendable to any other language or dataset. 
Pipeline supports different modalities: image, audio and video. . In our experiments, we focus on [MERA datasets](https://github.com/MERA-Evaluation/MERA), 
however, the presented pipeline can be generalized to other languages. The system is a set of models and Python scripts in a GitHub repository. 
We support two major functionalities for image, audio and video modalities: inference of membership detection model and training pipeline for new datasets.

## Usage
### Data
For start working we should convert our dataset into pandas format with following structure:

| question | answer | audio | ds_name  |
|----------|--------|-------|----------|

* `question` example:

```text
Помогите мне, пожалуйста.

Есть задача такого типа. Задача на понимание музыки и невербальных аудио сигналов.

Имеется 1 аудиофайл

Аудиофайл: <audio>
Вопрос:
Сколько раз слышен сильный всплеск воды?

A. 10
B. 4
C. 12
D. 8

Определите ответ к задаче, учитывая, что первому из предложенных вариантов ответа присваивается литера А, второму литера B, третьему литера C и так далее по английскому алфавиту. В качестве ответа выведите, пожалуйста, литеру, соответствующую верному варианту ответа из предложенных. Финальный ответ прошу написать после слова ОТВЕТ (литера через пробел после этого слова).
```

* `answer` example: 'B'.
* `audio` - is the modality column. For video we should put `video`, for image - `image`.
* `ds_name` is the dataset name. For example `ruEnvAQA`.

### Train
All pipeline contains following steps:
1. SFT-Lora MLLM finetuning (if need)
2. Neighbor generation 
3. Embedding generation 
4. Loss computation
5. Attack model training
#### SFT-Lora MLLM finetuning
For finetuning run python commands.
##### Image
```bash
python job_launcher.py --script="smia.sft_finetune_image" \
  --train_df_path="path/to/train.csv" \
  --test_df_path="path/to/test.csv" \
  --num_train_epochs=5 \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --output_dir=f"data/models/sft/Qwen2.5-VL-3B-Instruct"
```
Here
* `train_df_path` - train dataset path
* `test_df_path` - test dataset path
* `model_id` - path to inital model
* `output_dir` - path for saving fintuning model

##### Video
```bash
python job_launcher.py --script="smia.video.train_qwen25vl" \
  --train_df_path="path/to/train.csv" \
  --test_df_path="path/to/test.csv" \
  --num_train_epochs=5 \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --output_dir=f"data/models/sft/Qwen2.5-VL-3B-Instruct"
```
##### Audio
```bash
python job_launcher.py --script="smia.audio.train_qwen2" \
  --train_df_path="path/to/train.csv" \
  --test_df_path="path/to/test.csv" \
  --num_train_epochs=5 \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --output_dir=f"data/models/sft/Qwen2.5-VL-3B-Instruct"
```
#### Neighbor generation 
```bash
python job_launcher.py --script="smia.neighbors" \
  --model_path="ai-forever/FRED-T5-1.7B" \
  --dataset_path="path/to/train.csv" \
  --max_text_len=4000
```
Here
* `model_path` - embedder model for masking neighbors generation
* `dataset_path` - path to dataset for generating neighbors
* `max_text_len` - max of text length in number of characters
#### Embedding generation
```bash
python job_launcher.py --script="smia.embeds_text_calc" \
  --embed_model="intfloat/e5-mistral-7b-instruct" \
  --df_path="path/to/train.csv"
```
Here
* `embed_model` - embedder path
* `df_path` - path to dataset for generating embeddings
#### Loss computation
##### Image
```bash
python job_launcher.py --script="smia.image.loss_calc" \
  --model_id=Qwen/Qwen2.5-VL-3B-Instruct \
  --model_name=Qwen2.5-VL-3B-Instruct \
  --label=0 \
  --df_path="path/to/train.csv"
```
Here
* `model_id` - path MLLM model
* `model_name` - name of MLLM model (using for store results)
* `label` - label of dataset `0` or `1`
* `df_path` - path to dataset for calculating loss
##### Audio
```bash
python job_launcher.py --script="smia.audio.loss_calc_qwen2" \
  --model_id=Qwen/Qwen2-Audio-7B-Instruct \
  --model_name=Qwen2-Audio-7B-Instruct \
  --label=0 \
  --df_path="path/to/train.csv"
```
##### Video
```bash
python job_launcher.py --script="smia.video.loss_calc_qwen25" \
  --model_id=Qwen/Qwen2.5-VL-3B-Instruct \
  --model_name=Qwen/Qwen2.5-VL-3B-Instruct \
  --label=0 \
  --df_path="path/to/train.csv"
```
#### Attack model training
Before training we need prepare data and merge all parts files embeddings and losses:
```bash
python job_launcher.py --script="smia.utils.mds_dataset" \
  --save_dir="path/to/save/mds/dataset" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --origin_df_path="path/to/train.csv" \
  --shuffle=0 \
  --labels="0,1" \
  --modality_key="video"
```
Here
* `save_dir` - path for saving merged dataset
* `model_name` - name of MLLM model (using for store results)
* `shuffle` - not shuffle data `0` or shuffle `1`
* `labels` - list of labels in dataset
* `modality_key` - modality column

After data preparation run trainin attack model neural network MSMIA:
```bash
python job_launcher.py --script="smia.train" \
  --train_dataset_path="train/mds/path" \
  --val_dataset_path="test/mds/path" \
  --model_name="SMIABaseLineModelLossNormSTDV2" \
  --output_dir="path/to/model/save" \
  --num_train_epochs=10 \
  --optim="adafactor" \
  --learning_rate=0.00005 \
  --max_grad_norm=10 \
  --warmup_ratio=0.03 \
  --sigmas_path="data/pd_datasets/video/sigmas.json" \
  --sigmas_type="std"
```
Here
* `train_dataset_path` - path to train mds dataset
* `val_dataset_path` - path to test mds dataset
* `model_name` - name MSMIA neural network architecture
* `num_train_epochs` - number of training epochs
* `output_dir` - path to save MSMIA model
* `optim` - pytorch optimizer name
* `learning_rate` - learning rate
* `max_grad_norm` - max gradient normalization
* `warmup_ratio` - warmup ratio for optimization
* `sigmas_path` - path for dict with normalization parameters
* `sigmas_type` - type of normalization

### Inference
For inference we repeat 2, 3, 4 steps from training stage:
2. Neighbor generation
3. Embedding generation
4. Loss computation

For inference MSMIA model on new data we should run command:
```bash
python job_launcher.py --script="smia.smia_inference" \
  --model_name="SMIABaseLineModelLossNormSTDV2" \
  --model_path="path/to/model/save" \
  --test_path="path/to/test.csv" \
  --save_path="path/to/save/predictions.csv" \
  --save_metrics_path="path/to/save/metrics" 
```
Here
* `model_name` - name MSMIA neural network architecture
* `model_path` - path to load MSMIA model
* `test_path` - path to test dataset
* `save_path` - path to save predictions
* `save_metrics_path` - path to save metrics

### Authors
* Emelyanov Anton
* Kudriashov Sergei
* Alena Fenogenova