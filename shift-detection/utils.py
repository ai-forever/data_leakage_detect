from datasets import load_dataset, concatenate_datasets
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from PIL import Image

def get_dataset(args, dataset_name, attack_name):
    print(dataset_name)
    if dataset_name == 'wikimia':
        dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length128")
        X = dataset['input']
        y = dataset['label']
        members = dataset.filter(lambda d: d['label']==1)['input']
        nonmembers = dataset.filter(lambda d: d['label']==0)['input']
    if dataset_name == 'wikimia-hard':
        dataset = load_dataset("hallisky/wikiMIA-2024-hard", split=f"test")
        X = dataset['input']
        y = dataset['label']
        members = dataset.filter(lambda d: d['label']==1)['input']
        nonmembers = dataset.filter(lambda d: d['label']==0)['input']
    if 'wikimia-24' in dataset_name :
        _, length_split = dataset_name.rsplit("-", 1)
        if length_split == "all":
            dataset = load_dataset("wjfu99/WikiMIA-24", ) 
            dataset = concatenate_datasets([*dataset.values()])
        else:
            dataset = load_dataset("wjfu99/WikiMIA-24", split=f"WikiMIA_length{length_split}")
        X = dataset['input']
        y = dataset['label']
        members = dataset.filter(lambda d: d['label']==1)['input']
        nonmembers = dataset.filter(lambda d: d['label']==0)['input']
    elif dataset_name == 'bookmia':
        dataset = load_dataset("swj0419/BookMIA")['train']
        # print(dataset[0])
        X = dataset['snippet']
        y = dataset['label']
        members = dataset.filter(lambda d: d['label']==1)['snippet']
        nonmembers = dataset.filter(lambda d: d['label']==0)['snippet']
    elif dataset_name == 'temporal_wiki':
        dataset = load_dataset("iamgroot42/mimir", "temporal_wiki", split="none")
        data_len = len(dataset)
        X = dataset['member']+dataset['nonmember']
        y = data_len*[1]+data_len*[0]
        members = dataset['member']
        nonmembers = dataset['nonmember']
    elif dataset_name == 'temporal_arxiv':
        dataset = load_dataset("iamgroot42/mimir", "temporal_arxiv", split="2021_06")
        data_len = len(dataset)
        X = dataset['member']+dataset['nonmember']
        y = data_len*[1]+data_len*[0]
        members = dataset['member']
        nonmembers = dataset['nonmember']
    elif dataset_name == 'arxiv_tection':
        dataset = load_dataset("avduarte333/arXivTection")['train']
        X = dataset['Example_A']
        y = dataset['Label']
        members = dataset.filter(lambda d: d['Label']==1)['Example_A']
        nonmembers = dataset.filter(lambda d: d['Label']==0)['Example_A']
    elif dataset_name == 'book_tection':
        dataset = load_dataset("avduarte333/BookTection")['train']
        filtered_dataset = dataset.filter(lambda d: d['Length']=='medium')
        print(len(dataset), len(filtered_dataset))
        X = dataset['Example_A']
        y = dataset['Label']
        members = dataset.filter(lambda d: d['Label']==1)['Example_A']
        nonmembers = dataset.filter(lambda d: d['Label']==0)['Example_A']
    elif dataset_name == 'arxiv_1m':
        members = []
        nonmembers = []
        member_files = glob.glob("data/arxiv1m/member/*.txt")
        nonmember_files = glob.glob("data/arxiv1m/nonmember/*.txt")
        print(len(member_files), len(nonmember_files))
        for m in member_files:
            f = open(m, "r")
            text = f.read()
            members.append(text)
            f.close()
        for m in nonmember_files:
            f = open(m, "r")
            text = f.read()
            nonmembers.append(text)
            f.close()
        X = members + nonmembers
        y = [1]*len(members) + [0]*len(nonmembers)
    elif dataset_name == 'arxiv1m_1m':
        members = np.load("data/arxiv1m_1m/member.npy")
        nonmembers = np.load("data/arxiv1m_1m/nonmember.npy")
        X = list(members) + list(nonmembers)
        y = [1]*len(members) + [0]*len(nonmembers)
    elif dataset_name == 'multi_web':
        f = open('data/multi_web/member.txt', 'r')
        members = f.read().split('\n')
        f = open('data/multi_web/nonmember.txt', 'r')
        nonmembers = f.read().split('\n')
        X = members + nonmembers
        y = [1]*len(members) + [0]*len(nonmembers)
    elif dataset_name == 'laion_mi':
        f = open('data/laion_mi/member.txt', 'r')
        members = f.read().split('\n')
        f = open('data/laion_mi/nonmember.txt', 'r')
        nonmembers = f.read().split('\n')
        X = members + nonmembers
        y = [1]*len(members) + [0]*len(nonmembers)
    elif dataset_name == 'laion_mi_image':
        dataset = pd.read_csv("data/laion_mi_image/laion_mi_image.csv")
        X = list(dataset['image_paths'])
        X = [Image.open(image_path).convert("RGB") for image_path in X]
        y = list(dataset['label'])
        members = list(dataset[dataset["label"] == 1]["image_paths"])
        nonmembers = list(dataset[dataset["label"] == 0]["image_paths"])
    elif dataset_name == 'gutenberg':
        members = []
        nonmembers = []
        member_files = glob.glob("data/gutenberg/member/*.txt")
        nonmember_files = glob.glob("data/gutenberg/nonmember/*.txt")
        # print(len(member_files), len(nonmember_files))
        for m in member_files:
            f = open(m, "r")
            text = f.read()
            members.append(text)
            f.close()
        for m in nonmember_files:
            f = open(m, "r")
            text = f.read()
            nonmembers.append(text)
            f.close()
        X = members + nonmembers
        y = [1]*len(members) + [0]*len(nonmembers)
    elif dataset_name.startswith('vl_mia_text'):
        *_, split_length = dataset_name.split("_")
        if split_length == "all":
            dataset = load_dataset("JaineLi/VL-MIA-text", "llava_v15_gpt_text",)
            dataset = concatenate_datasets([*dataset.values()])
        else:
            dataset = load_dataset("JaineLi/VL-MIA-text", "llava_v15_gpt_text", split=f"length_{split_length}")
        X = list(dataset['input'])
        y = list(dataset['label'])
        members = list(dataset.filter(lambda d: d['label']==1)['input'])
        nonmembers = list(dataset.filter(lambda d: d['label']==0)['input'])
    elif dataset_name.startswith('vl_mia_img'):
        *_, subset = dataset_name.split("_", 2)
        dataset = load_dataset("JaineLi/VL-MIA-image", subset, split="train",)
        X = list(dataset['image'])
        y = list(dataset['label'])
        members = list(dataset.filter(lambda d: d['label']==1)['image'])
        nonmembers = list(dataset.filter(lambda d: d['label']==0)['image'])
    elif dataset_name == 'custom':
        # Handle custom dataset
        if not args.custom_data_path:
            raise ValueError("Custom dataset path must be provided with --custom_data_path")
        if not args.custom_feature_column:
            raise ValueError("Feature column name must be provided with --custom_feature_column")
        if not args.custom_label_column:
            raise ValueError("Label column name must be provided with --custom_label_column")
        
        # Load the custom dataset
        dataset = pd.read_csv(args.custom_data_path)
        
        # Extract features and labels
        X = list(dataset[args.custom_feature_column])
        X = [Image.open(image).convert("RGB") for image in X] if args.attack == "bag_of_visual_words" else None
        y = list(dataset[args.custom_label_column])
        
        # Split into members and non-members based on labels
        members = list(dataset[dataset[args.custom_label_column] == 1][args.custom_feature_column])
        nonmembers = list(dataset[dataset[args.custom_label_column] == 0][args.custom_feature_column])

    return X, y, members, nonmembers

def get_roc_auc(y_true, y_pred_proba):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba) 
    return metrics.auc(fpr, tpr)

def get_tpr_metric(y_true, y_pred_proba, fpr_budget):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba) 
    tpr_at_low_fpr = np.interp(fpr_budget/100, fpr,tpr)
    return tpr_at_low_fpr

def plot_tpr_fpr_curve(y_true, y_pred_proba, fpr_budget):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba) 
    roc_auc = metrics.auc(fpr, tpr)
    tpr_at_low_fpr = np.interp(fpr_budget/100, fpr,tpr)
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.plot(fpr_budget/100, tpr_at_low_fpr, marker="x", markersize=10, markeredgecolor="red", markerfacecolor="green")
    plt.plot(fpr_budget/100, 0, marker="o", markersize=5, markerfacecolor="red", markeredgecolor="red")
    plt.vlines(fpr_budget/100, 0, tpr_at_low_fpr, color='r', linestyles='dashed')
    plt.hlines(tpr_at_low_fpr, 0, fpr_budget/100, color='r', linestyles='dashed')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


    