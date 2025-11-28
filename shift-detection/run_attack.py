import argparse
from date_detection import date_detection_arxiv, date_detection_basic
from bag_of_words import bag_of_words_basic
from greedy_selection import greedy_selection_arxiv, greedy_selection_basic, greedy_selection_wiki
from bag_of_visual_feats import bag_of_visual_words_basic
from bag_of_audio_feats import bag_of_audio_words_basic
from utils import *


if __name__ == "__main__":
    np.random.seed(0) # For reproducibility

    parser = argparse.ArgumentParser(description='Blind Attacks on Membership Inference Attack Evaluation Datasets')
    parser.add_argument('--dataset', help='dataset name', 
        choices=[
            'wikimia', 'wikimia-hard',
            'wikimia-24-all', 'wikimia-24-32', 
            'wikimia-24-64', 'wikimia-24-128', 'wikimia-24-256',
            'bookmia', 'temporal_wiki', 
            'temporal_arxiv', 'arxiv_tection', 
            'book_tection', 'arxiv_1m', 'arxiv1m_1m', 
            'multi_web', 'gutenberg', 
            'laion_mi', 'laion_mi_image',
            'vl_mia_text_4', 'vl_mia_text_16', 'vl_mia_text_32', 'vl_mia_text_64',
            'vl_mia_img_Flickr', 'vl_mia_img_Flickr_2k', 'vl_mia_img_Flickr_10k', 'vl_mia_img_dalle',
            'custom'
            ], default='bookmia')
    parser.add_argument('--custom_data_path', help='path to custom dataset CSV file')
    parser.add_argument('--custom_feature_column', help='column name for features in custom dataset')
    parser.add_argument('--custom_label_column', help='column name for labels in custom dataset')
    parser.add_argument('--attack', help='attack method', choices=['date_detection','bag_of_words','greedy_selection', 'bag_of_visual_words', 'bag_of_audio_words'], default='bag_of_words')
    parser.add_argument('--plot_roc', help='set to plot FPR vs TPR curve', action="store_true")
    parser.add_argument('--hypersearch', help='set to redo hyperparam search instead of using default params', action="store_true")
    parser.add_argument('--fpr_budget', help='x for computing TPR@x%FPR', type=float, default=1)

    args = parser.parse_args()
    print(args.dataset, args.attack)

    X, y, members, nonmembers = get_dataset(args, args.dataset, args.attack)
    dataset_name = args.dataset if not args.dataset.startswith("vl_mia_") else args.dataset.rsplit("_", 1)[0]

    if args.attack == 'date_detection':
        if dataset_name == 'arxiv_1m':
            date_detection_arxiv(X, y, members, nonmembers, dataset_name=dataset_name, fpr_budget=args.fpr_budget, plot_roc=args.plot_roc)
        else:
            date_detection_basic(X,y, dataset_name=dataset_name, fpr_budget=args.fpr_budget, plot_roc=args.plot_roc)
    elif args.attack == 'bag_of_words':
        bag_of_words_basic(X,y, dataset_name=dataset_name, fpr_budget=args.fpr_budget, plot_roc=args.plot_roc, hypersearch=args.hypersearch)
    elif args.attack == 'greedy_selection':
        if dataset_name == 'temporal_wiki':
            greedy_selection_wiki(members, nonmembers, dataset_name, args.fpr_budget, args.plot_roc)
        elif dataset_name == 'arxiv1m_1m':
            greedy_selection_arxiv(members, nonmembers, dataset_name, args.fpr_budget, args.plot_roc)
        else:
            greedy_selection_basic(members, nonmembers, dataset_name, args.fpr_budget, args.plot_roc)
    elif args.attack == "bag_of_visual_words":
        bag_of_visual_words_basic(X,y, dataset_name=dataset_name, fpr_budget=args.fpr_budget, plot_roc=args.plot_roc, hypersearch=args.hypersearch)
    elif args.attack == "bag_of_audio_words":
        bag_of_audio_words_basic(X,y, dataset_name=dataset_name, fpr_budget=args.fpr_budget, plot_roc=args.plot_roc, hypersearch=args.hypersearch)
