# Blind multimodal membership inference attacks for baseline MIA evaluation

### Description

An extension of work by [Das et al., 2024](https://arxiv.org/abs/2406.16201) to multimodal data. We provide blind baselines for text, image and audio datasets and encourage researchers to use them in their MIA evaluations and benchmark creation.

## How to run our attacks?
At the root of the repository, run the following to install required dependencies:

```pip install -r requirements.txt```

### Setting up Datasets

All datasets except for the Arxiv (1 month vs 1 month) and LAION-MI datasets are accessible from the [original repository](https://github.com/ethz-spylab/Blind-MIA). 
For arxiv1m_1m dataset, check [this section](#setting-up-the-arxiv-1-month-vs-1-month-dataset)
For LAION-MI dataset, check [this section](#setting-up-laion-mi-images)

### Run Attacks

Run the ``run_attack.py`` script with the required command line arguments using the command below.

``` python3 run_attack.py --dataset <dataset> --attack <attack> ``` 

where ```<dataset>``` is one of the datasets from the following list:

```'wikimia', 'wikimia-hard', 'wikimia-24-all', 'wikimia-24-32', 'wikimia-24-64', 'wikimia-24-128', 'wikimia-24-256', 'bookmia', 'temporal_wiki', 'temporal_arxiv', 'arxiv_1m', 'arxiv_1m_1m', 'multi_web',  'gutenberg', 'laion_mi', 'laion_mi_image', 'vl_mia_text_4', 'vl_mia_text_16', 'vl_mia_text_32', 'vl_mia_text_64', 'vl_mia_img_Flickr', 'vl_mia_img_Flickr_2k', 'vl_mia_img_Flickr_10k', 'vl_mia_img_dalle', 'custom' ```

and ```<attack>``` is one of the following attacks:

1. ``date_detection``: Applicable for temporal datasets ``wikimia``, ```temporal_wiki```, ```temporal_arxiv```, ``arxiv1m``, and ```arxiv1m_1m```. It infers membership based on dates extracted from the text.
2. ``bag_of_words``: Applicable for all datasets. It infers membership based on the bag-of-words representation of the text.
3. ``greedy_selection``: Applicable for all datasets but works more efficiently on datasets with shorter text samples. Gives best results on datasets: ``temporal_wiki, arxiv1m_1m, multi_web, laion_mi``
4. ``bag_of_visual_words``: Applicable to image datasets. Infers distribution shift from image statistics, such as SIFT, DCT, color and local binary patters.
5. ``bag_of_audio_words``: Applicable to audio datasets. Infers distribution shift from audio sample statistics, e.g. spectral features.

### Example:
For example, to run the bag-of-words attack on the WikiMIA dataset, run the following command:

``` python3 run_attack.py --dataset WikiMIA --attack bag_of_words ```
### Optional Flags:
To specify the FPR budget to be used to compute the TPR@x%FPR, use the ``fpr_budget`` flag and specify the desired FPR budget. For example, to compute the TPR@5%FPR, run the following command:

``` python3 run_attack.py --dataset WikiMIA --attack bag_of_words --fpr_budget 5 ```

To redo the hyper-parameter search, add the flag ``--hypersearch``, otherwise the bag of words attack uses the best default hyper-parameters. To plot the AUC ROC curve, add the flag ``--plot_roc``. 
### Using Custom Datasets

To use a custom dataset, specify `--dataset custom` along with the following additional arguments:
- `--custom_data_path`: Path to your CSV file containing the dataset
- `--custom_feature_column`: Name of the column containing the text features
- `--custom_label_column`: Name of the column containing the labels (should be 1 for members, 0 for non-members)

Example:
``` python3 run_attack.py --dataset custom --custom_data_path /path/to/your/dataset.csv --custom_feature_column text --custom_label_column label --attack bag_of_words ```

Note: Your custom dataset must be in CSV format with at least two columns: one for features (text) and one for binary labels (1 for members, 0 for non-members).


## Results

| MI Dataset           | Metric                    | Best Attack | Ours | Blind Attack Type |
|----------------------|---------------------------|-------------|------|:-------------------:|
|                      | <span style="color:cyan"> *Temporal Shifted Datasets* </span> |             |      |                   |
| **WikiMIA**          | TPR@5%FPR                 |        43.2 | 94.7 | ``bag_of_words``               |
|                      | AUCROC                    |        83.9 |   99 | ``bag_of_words``               |
| **WikiMIA-24**       | TPR@1%FPR                 |             | 98.3 | ``bag_of_words``               |
|                      | AUCROC                    |        99.8 | 99.9 | ``bag_of_words``               |
| **WikiMIA-Hard**     | TPR@1%FPR                 |             | 3.67 | ``bag_of_words``               |
|                      | AUCROC                    |        64.0 | 57.7 | ``bag_of_words``               |
| **BookMIA**          | TPR@5%FPR                 |        33.6 | 64.5 | ``bag_of_words``               |
|                      | AUCROC                    |          88 | 91.4 | ``bag_of_words``               |
| **Temporal Wiki**    | TPR@1%FPR                 |             | 36.5 | ``greedy_selection``           |
|                      | AUCROC                    |        79.6 | 79.9 | ``greedy_selection``           |
| **Temporal Arxiv**   | TPR@1%FPR                 |             |  9.1 | ``bag_of_words``               |
|                      | AUCROC                    |        74.5 | 75.3 | ``bag_of_words``               |
| **Arxiv**            | TPR@1%FPR                 |         5.9 | 10.6 | ``date_detection``             |
| (all vs 1 month)     | AUCROC                    |        67.8 | 72.3 | ``date_detection``             |
| **Arxiv**            | TPR@1%FPR                 |         2.5 |  2.7 | ``greedy_selection``           |
| (1 month vs 1 month) |                           |             |      |                   |
| **VL-MIA Text**
|*Length 32*           | TPR@5%FPR                 |             |      | ``bag_of_words``               |
|                      | AUCROC                    |        96.2 | 84.9 | ``bag_of_words``               |
|*Length 64*           | TPR@5%FPR                 |             |      | ``bag_of_words``               |
|                      | AUCROC                    |        99.3 | 95.5 | ``bag_of_words``               |
|                      | <span style="color:cyan"> *Image datasets with distribution shifts* </span>        |             |      |                   |
| **VL-MIA Images**    | TPR@5%FPR                 |        24.7 | 95.0 | ``bag_of_visual_words``               |
| *Flickr*             | AUCROC                    |        71.3 | 99.0 | ``bag_of_visual_words``               |
| **VL-MIA Images**    | TPR@5%FPR                 |       22.0  | 99.6 | ``bag_of_visual_words``               |
| *Dalle*              | AUCROC                    |        70.7 | 99.9 | ``bag_of_visual_words``               |
| **LAION-MI Images**  | TPR@1%FPR                 |       2.42  | 1.11 | ``bag_of_visual_words``               |
|                      | AUCROC                    |             | 52.2 | ``bag_of_visual_words``               |
|                      | <span style="color:cyan"> *Biased Replication* </span>        |             |      |                   |
| **LAION-MI Captions**| TPR@1%FPR                 |         2.5 |  8.9 | ``greedy_selection``            |
| **Gutenberg**        | TPR@1%FPR                 |        18.8 | 55.1 | ``greedy_selection``            |
|                      | AUCROC                    |        85.6 | 96.1 | ``bag_of_words``               |

### Setting up the Arxiv (1 month vs 1 month) dataset

We handle this dataset separately because it is too big to push to the repository. Here are trhe steps to extract the dataset:
1. Download the whole arxiv dataset from [here]( https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-arxiv-refine-result.jsonl).
2. Run the data extracion script which will save the processed dataset in the [arxiv1m_1m](data/arxiv1m_1m) folder. 

``` python3 data_script_1m_1m.py --path <path to the downloaded jsonl file>```

3. Run the attack on the dataset using the command below:

``` python3 run_attack.py --dataset arxiv1m_1m --attack greedy_selection ```

### Setting up LAION-MI images

This dataset is also handled separately. However, we found that a huge amount of images from [laion_mi](https://huggingface.co/datasets/antoniaaa/laion_mi) are already unavailable from predefined urls. 
Thus results may vary depending on the amount of images accessible through the provided links. Steps to obtain images for evaluation:

1. Run the script to save images and the dataset for evaluation. 

``` python3 data_script_laion_mi.py```

2. Run the attack on the dataset using the command below:

``` python run_attack.py --dataset laion_mi_image  --fpr_budget 1 --attack 'bag_of_visual_words --hypersearch```


### Acknowledgements

```bibtex
@misc{das2024blindbaselinesbeatmembership,
      title={Blind Baselines Beat Membership Inference Attacks for Foundation Models}, 
      author={Debeshee Das and Jie Zhang and Florian Tramèr},
      year={2024},
      eprint={2406.16201},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2406.16201}, 
}
```
