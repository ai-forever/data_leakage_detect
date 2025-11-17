from statistics import mean, pstdev
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from utils import *  

def train_classifier(X_train, X_test, y_train, y_test, fpr_budget, plot_roc, 
                     params={'max_features': 30, 'vectorizer': 'tf', 'model_type': 'stack'}):
    """Train and evaluate a classifier."""
    if params['vectorizer'] == 'tf':
        converter = TfidfVectorizer(max_features=params['max_features'])
    elif params['vectorizer'] == 'count':
        converter = CountVectorizer(max_features=params['max_features'])

    X_train_Tfidf = converter.fit_transform(X_train).toarray()
    X_test_Tfidf = converter.transform(X_test).toarray()

    # Model selection
    if params['model_type'] == 'random':
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    elif params['model_type'] == 'gaussian':
        model = GaussianNB()
    elif params['model_type'] == 'multi':
        model = MultinomialNB()
    elif params['model_type'] == 'gradient':
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=42)
    elif params['model_type'] == 'stack':
        estimators = [
            ('random', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
            ('gaussian', GaussianNB()),
            ('multi', MultinomialNB()),
            ('gradient', GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=5, random_state=42))
        ]
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42))

    model.fit(X_train_Tfidf, y_train)
    y_pred_proba = model.predict_proba(X_test_Tfidf)[:, 1]

    roc_auc = get_roc_auc(y_test, y_pred_proba)
    tpr_at_low_fpr = get_tpr_metric(y_test, y_pred_proba, fpr_budget)

    if plot_roc:
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"TPR@{fpr_budget}%FPR: {tpr_at_low_fpr:.4f}")
        plot_tpr_fpr_curve(y_test, y_pred_proba, fpr_budget)
    else:
        return roc_auc, tpr_at_low_fpr

def evaluate_cross_val(X, y, fpr_budget, params, n_splits=5, random_state=42):
    """Evaluate model using stratified K-fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores, tpr_scores = [], []

    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        roc_auc, tpr = train_classifier(X_train, X_test, y_train, y_test, fpr_budget, plot_roc=False, params=params)
        auc_scores.append(roc_auc)
        tpr_scores.append(tpr)

    return auc_scores, tpr_scores

def evaluate_random_class_splits(X, y, fpr_budget, params, n_splits=5, random_state=42):
    """Evaluate model on random splits of the negative class (0)."""
    np.random.seed(random_state)
    y_random = y.copy()
    y_random = np.random.permutation(y_random)  # Shuffle class labels

    return evaluate_cross_val(X, y_random, fpr_budget, params, n_splits, random_state)

def hyperparam_search(X, y, fpr_budget, n_splits=5):
    """Grid search with cross-validation."""
    param_grid = {
        'max_features': [10, 15, 20, 30, 40, 50, 60],
        'vectorizer': ['tf', 'count'],
        'model_type': ['multi', 'gaussian', 'random', 'gradient', 'stack']
    }
    best_score = -1
    best_params = None

    for max_features in param_grid['max_features']:
        for vectorizer in param_grid['vectorizer']:
            for model_type in param_grid['model_type']:
                params = {'max_features': max_features, 'vectorizer': vectorizer, 'model_type': model_type}
                auc_scores, _ = evaluate_cross_val(X, y, fpr_budget, params, n_splits)
                mean_auc = mean(auc_scores)
                if mean_auc > best_score:
                    best_score = mean_auc
                    best_params = params
    return best_params

def bag_of_words_basic(X, y, dataset_name, fpr_budget, plot_roc=False, hypersearch=False):
    """Main function with CV and random-split comparison."""
    default_params = {
        'wikimia': {'max_features': 34, 'vectorizer': 'tf', 'model_type': 'gaussian'},
        'bookmia': {'max_features': 58, 'vectorizer': 'count', 'model_type': 'stack'},
        'temporal_wiki': {'max_features': 52, 'vectorizer': 'tf', 'model_type': 'stack'},
        'temporal_arxiv': {'max_features': 62, 'vectorizer': 'count', 'model_type': 'stack'},
        'arxiv_tection': {'max_features': 62, 'vectorizer': 'tf', 'model_type': 'stack'},
        'book_tection': {'max_features': 54, 'vectorizer': 'tf', 'model_type': 'stack'},
        'arxiv_1m': {'max_features': None, 'vectorizer': 'tf', 'model_type': 'stack'},
        'arxiv1m_1m': {'max_features': 52, 'vectorizer': 'tf', 'model_type': 'gaussian'},
        'multi_web': {'max_features': 38, 'vectorizer': 'count', 'model_type': 'gradient'},
        'laion_mi': {'max_features': 10, 'vectorizer': 'tf', 'model_type': 'gaussian'},
        'gutenberg': {'max_features': None, 'vectorizer': 'tf', 'model_type': 'multi'},
        #'vl_mia_text': {'max_features': None, 'vectorizer': 'tf', 'model_type': 'multi'},
    }

    X, y = np.array(X), np.array(y)

    params = hyperparam_search(X, y, fpr_budget) if hypersearch else default_params.get(dataset_name, 
        {'max_features': 62, 'vectorizer': 'tf', 'model_type': 'stack'}
        )
    print("Best params:", params)

    # Evaluate on original task
    auc_scores, tpr_scores = evaluate_cross_val(X, y, fpr_budget, params)
    print(f"Original Task (AUC): {mean(auc_scores):.4f} ± {pstdev(auc_scores):.4f}")
    print(f"Original Task (TPR@{fpr_budget}%FPR): {mean(tpr_scores):.4f} ± {pstdev(tpr_scores):.4f}")

    # Evaluate on random-split task (negative class only)
    rand_auc, rand_tpr = evaluate_random_class_splits(X, y, fpr_budget, params)
    print(f"Random-Split Task (AUC): {mean(rand_auc):.4f} ± {pstdev(rand_auc):.4f}")
    print(f"Random-Split Task (TPR@{fpr_budget}%FPR): {mean(rand_tpr):.4f} ± {pstdev(rand_tpr):.4f}")

    if plot_roc:
        # Plot ROC for the first CV fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(skf.split(X, y))
        train_classifier(X[train_idx], X[test_idx], y[train_idx], y[test_idx], fpr_budget, plot_roc=True, params=params)