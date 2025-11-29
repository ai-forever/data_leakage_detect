# FiMMIA: scaling semantic perturbation-based membership inference across modalities

<p align="center">
  <picture>
    <img alt="FiMMIA" src="docs/FiMMIA_system_overview.png" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://github.com/ai-forever/data_leakage_detect/releases">
    <img alt="Release" src="https://img.shields.io/badge/release-v1.0.0-blue">
    </a>

</p>

This repository contains an implementation of **F**i**MMIA** - a modular **F**ramework for **M**ultimodal **M**embership **I**nference **A**ttacks (FiMMIA)

## Description
The system is the first collection of models and pipelines for membership inference attacks against multimodal large language models, built initially with a priority for the Russian language, and extendable to any other language or dataset. 
Pipeline supports different modalities: image, audio and video. In our experiments, we focus on [MERA datasets](https://github.com/MERA-Evaluation/MERA), however, the presented pipeline can be generalized to other languages. The system is a set of models and Python scripts in a GitHub repository. 

We support two major functionalities for image, audio and video modalities: inference of membership detection model and training pipeline for new datasets.

Pretrained models available on 🤗 HuggingFace [FiMMIA collection](https://huggingface.co/collections/ai-forever/fimmia).
## Distribution shift detection

Additionally, in [shift-detection](./shift-detection/) we release baseline attacks for multimodal data, tailored for distribution shift detection on target MIA datasets. Evaluation results as well as scripts for known datasets are provided in the respective folder. 

We encourage the community to run these baselines on their MIA benchmarks prior to their release or new methods evaluations to ensure fair and credible results.

We are grateful to [Das et al., 2024](https://arxiv.org/abs/2406.16201) for the initial text pipelines that has served as a base of this tool.

## Usage
The inference pipeline is shown at image below.

<p align="center">
  <picture>
    <img alt="FiMMIA Inference" src="docs/FiMMIA_Inference.png" style="max-width: 100%;">
  </picture>
</p>

### Data
For start working we should convert our dataset into pandas format with following structure:

| input | answer | audio | ds_name  |
|----------|--------|-------|----------|

* `input` example:

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
Whole pipeline contains the following steps:
1. SFT-Lora MLLM finetuning (if need)
2. Neighbor generation 
3. Embedding generation 
4. Loss computation
5. Attack model training
#### SFT-Lora MLLM finetuning
For finetuning run python commands.
##### Image
```bash
python job_launcher.py --script="fimmia.sft_finetune_image" \
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
python job_launcher.py --script="fimmia.video.train_qwen25vl" \
  --train_df_path="path/to/train.csv" \
  --test_df_path="path/to/test.csv" \
  --num_train_epochs=5 \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --output_dir=f"data/models/sft/Qwen2.5-VL-3B-Instruct"
```
##### Audio
```bash
python job_launcher.py --script="fimmia.audio.train_qwen2" \
  --train_df_path="path/to/train.csv" \
  --test_df_path="path/to/test.csv" \
  --num_train_epochs=5 \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --output_dir=f"data/models/sft/Qwen2.5-VL-3B-Instruct"
```
#### Neighbor generation 
```bash
python job_launcher.py --script="fimmia.neighbors" \
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
python job_launcher.py --script="fimmia.embeds_text_calc" \
  --embed_model="intfloat/e5-mistral-7b-instruct" \
  --df_path="path/to/train.csv" \
  --part_size=5000
```
Here
* `embed_model` - embedder path
* `df_path` - path to dataset for generating embeddings
* `part_size` - lines for split dataframe into smaller frames
#### Loss computation
##### Image
```bash
python job_launcher.py --script="fimmia.image.loss_calc" \
  --model_id=Qwen/Qwen2.5-VL-3B-Instruct \
  --model_name=Qwen2.5-VL-3B-Instruct \
  --label=0 \
  --df_path="path/to/train.csv" \
  --part_size=5000
```
Here
* `model_id` - path MLLM model
* `model_name` - name of MLLM model (using for store results)
* `label` - label of dataset `0` or `1`
* `df_path` - path to dataset for calculating loss
* `part_size` - lines for split dataframe into smaller frames
##### Audio
```bash
python job_launcher.py --script="fimmia.audio.loss_calc_qwen2" \
  --model_id=Qwen/Qwen2-Audio-7B-Instruct \
  --model_name=Qwen2-Audio-7B-Instruct \
  --label=0 \
  --df_path="path/to/train.csv" \
  --part_size=5000
```
##### Video
```bash
python job_launcher.py --script="fimmia.video.loss_calc_qwen25" \
  --model_id=Qwen/Qwen2.5-VL-3B-Instruct \
  --model_name=Qwen/Qwen2.5-VL-3B-Instruct \
  --label=0 \
  --df_path="path/to/train.csv" \
  --part_size=5000
```
#### Attack model training
Before training we need prepare data and merge all parts of files containing embeddings and losses:
```bash
python job_launcher.py --script="fimmia.utils.mds_dataset" \
  --save_dir="path/to/save/mds/dataset" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --origin_df_path="path/to/train.csv" \
  --shuffle=0 \
  --labels="0,1" \
  --modality_key="video" \
  --single_file=1
```
Here
* `save_dir` - path for saving merged dataset
* `model_name` - name of MLLM model (using for store results)
* `shuffle` - not shuffle data `0` or shuffle `1`
* `labels` - list of labels in dataset
* `modality_key` - modality column
* `single_file` - run on single file or batches

After data preparation run training of an attack model neural network FiMMIA:
```bash
python job_launcher.py --script="fimmia.train" \
  --train_dataset_path="train/mds/path" \
  --val_dataset_path="test/mds/path" \
  --model_name="FiMMIABaseLineModelLossNormSTDV2" \
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
* `model_name` - name FiMMIA neural network architecture
* `num_train_epochs` - number of training epochs
* `output_dir` - path to save FiMMIA model
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

For inference FiMMIA model on new data we should run command:
```bash
python job_launcher.py --script="fimmia.fimmia_inference" \
  --model_name="FiMMIABaseLineModelLossNormSTDV2" \
  --model_path="path/to/model/save" \
  --test_path="test/mds/path" \
  --save_path="path/to/save/predictions.csv" \
  --save_metrics_path="path/to/save/metrics" 
```
Here
* `model_name` - name FiMMIA neural network architecture
* `model_path` - path to load FiMMIA model
* `test_path` - path to test dataset
* `save_path` - path to save predictions
* `save_metrics_path` - path to save metrics

### Gradient attribution

We also support running an gradient-based feature attribution on the FiMMIA model, intended to calculate a relative impact of loss and embedding related parts. 
The pipeline saves results and provides an option to draw graphs of attrbution metrics. The results are saved into the same folder as an FiMMIA model.

To run attribution:
```bash
python job_launcher.py --script="fimmia.attribute_fimmia" \
  --model_dir="path/to/fimmia_model_folder" \
  --mds_dataset_path="path/to/mds_dataset_folder" \
  --model_cls="BaseLineModelV2" \
  --embedding_size=1024 \
  --modality_embedding_size=1024 \
  --add_attribution_noise=False \
  --create_graphs=True
```
Here
* `model_cls` - name of a FiMMIA neural network architecture
* `model_dir` - path to load FiMMIA model
* `mds_dataset_path` - path to the dataset to attribute
* `embedding_size` - dimension of the embedding input
* `modality_embedding_size` - dimension of the modality embedding input (only used in case modal embeddings are used)
* `add_attribution_noise` - whether to use stochastic perturbations (e.g. NoiseTunnel) to enhance reliability of the method
* `create_graphs` - whether to create graphs of attribution results


### Authors
* Emelyanov Anton
* Kudriashov Sergei
* Alena Fenogenova

