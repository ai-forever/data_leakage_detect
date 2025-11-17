import cv2
import operator
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
from sklearn.cluster import KMeans
from statistics import mean, pstdev
from utils import *  

def extract_sift_features(images, max_features=50):
    """Extract SIFT descriptors and build a Bag of Visual Words (BoVW)."""
    sift = cv2.SIFT_create()
    descriptors_list = []
    
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, descriptors = sift.detectAndCompute(img_gray, None)
        descriptors_list.append(descriptors)
    
    # Cluster descriptors to build visual vocabulary
    all_descriptors = np.vstack([descriptor for descriptor in descriptors_list if descriptor is not None])
    kmeans = KMeans(n_clusters=max_features, random_state=42)
    kmeans.fit(all_descriptors)
    
    # Quantize descriptors for each image
    bovw_features = []
    for descriptors in descriptors_list:
        hist = np.zeros(max_features)
        if descriptors is not None:
            visual_words = kmeans.predict(descriptors)
            for word in visual_words:
                hist[word] += 1
        bovw_features.append(hist)
    
    return np.array(bovw_features)


def extract_dct_features(images, n_coeffs=20):
    """Extract DCT-based frequency-domain features."""
    dct_features = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        dct_coeffs = dct(dct(img_gray.T, norm='ortho').T, norm='ortho')
        coeffs_flat = dct_coeffs.flatten()
        top_coeffs = np.concatenate([coeffs_flat[:n_coeffs], coeffs_flat[-n_coeffs:]])
        dct_features.append(top_coeffs)
    return np.array(dct_features)


def extract_color_histograms(images, bins=32):
    """Extract RGB/HSV histograms."""
    hist_features = []
    for img in images:
        if len(img.shape) == 2:  # Grayscale
            hist = np.histogram(img, bins=bins, range=(0, 256))[0]
        else:  # Color (RGB or BGR)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist_h = np.histogram(hsv[:,:,0], bins=bins, range=(0, 180))[0]
            hist_s = np.histogram(hsv[:,:,1], bins=bins, range=(0, 256))[0]
            hist_v = np.histogram(hsv[:,:,2], bins=bins, range=(0, 256))[0]
            hist = np.concatenate([hist_h, hist_s, hist_v])
        hist_features.append(hist)
    return np.array(hist_features)


def extract_lbp_features(images, radius=1, n_points=8):
    """Extract Local Binary Pattern (LBP) features."""
    lbp_features = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        hist = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2))[0]
        lbp_features.append(hist)
    return np.array(lbp_features)


def extract_all_features(images, sift_clusters=50, dct_coeffs=20, color_bins=32, lbp_points=8):
    """Extract all features and concatenate them into a single vector per image."""
    sift_features = extract_sift_features(images, max_features=sift_clusters)
    dct_features = extract_dct_features(images, n_coeffs=dct_coeffs)
    lbp_features = extract_lbp_features(images, n_points=lbp_points)
    color_features = extract_color_histograms(images, bins=color_bins)

    print(sift_features.shape, dct_features.shape, lbp_features.shape, color_features.shape)
    
    max_sift_dim = sift_clusters
    if sift_features.shape[1] < max_sift_dim:
        pad_width = [(0, 0), (0, max_sift_dim - sift_features.shape[1])]
        sift_features = np.pad(sift_features, pad_width, mode='constant')
    
    features = np.hstack([
        sift_features,
        dct_features,
        color_features,
        lbp_features
    ])
    return features


def reduce_dimensionality(features, n_components=100):
    """Apply PCA to reduce feature dimensions while preserving 95% variance."""
    
    n_components = min(n_components, features.shape[1])
    
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    scaler = StandardScaler()
    pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
    reduced_features = pipeline.fit_transform(features)
    
    return reduced_features, pipeline

def evaluate_image_classifier(images, y, params, fpr_budget, n_splits=5, reduce_dim=True):
    """Train/evaluate a classifier on combined image features with CV."""

    X = extract_all_features(
        images,
        sift_clusters=params.get('sift_clusters', 50),
        dct_coeffs=params.get('dct_coeffs', 20),
        color_bins=params.get('color_bins', 32),
        lbp_points=params.get('lbp_points', 8)
    )
    
    if reduce_dim:
        X, _ = reduce_dimensionality(X, n_components=params.get('n_components', 100))
    
    model = get_model(params['model_type'])  
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    roc_auc_scores, tpr_at_low_fpr_scores = [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = operator.itemgetter(*train_idx)(X), operator.itemgetter(*test_idx)(X)
        y_train, y_test = operator.itemgetter(*train_idx)(y), operator.itemgetter(*test_idx)(y)

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = get_roc_auc(y_test, y_pred_proba)
        tpr_at_low_fpr = get_tpr_metric(y_test, y_pred_proba, fpr_budget)
        roc_auc_scores.append(roc_auc)
        tpr_at_low_fpr_scores.append(tpr_at_low_fpr)  
        
    return roc_auc_scores, tpr_at_low_fpr_scores


def get_model(model_type):
    """Return a classifier based on model_type."""
    if model_type == 'random':
        return RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    elif model_type == 'gradient':
        return GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=42)
    elif model_type == 'gaussian':
        return GaussianNB()
    elif model_type == 'stack':
        estimators = [
            ('random', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
            ('gaussian', GaussianNB()),
            ('gradient', GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=5, random_state=42))
        ]
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42))


def hyperparam_search(images, y, param_grid, fpr_budget, n_splits=5):
    """Grid search over feature extraction and model hyperparameters."""
    best_auc = -1
    best_params = None
    
    for sift_clusters in param_grid.get('sift_clusters', [50]):
        for model_type in param_grid.get('model_type', ['random']):
            for n_components in param_grid.get('n_components', [100]):
                params = {
                'sift_clusters': sift_clusters,
                'model_type': model_type,
                'n_components': n_components,
                }
            auc, fpr = evaluate_image_classifier(images, y, params, fpr_budget=fpr_budget, n_splits=n_splits)
            if (mean_auc := mean(auc)) > best_auc:
                best_auc = mean_auc
                best_params = params
    return best_params, best_auc


def evaluate_random_labels(images, y, params, fpr_budget, n_splits=5, random_state=42, reduce_dim=True):
    """Evaluate on randomly shuffled labels (negative class only)."""
    
    np.random.seed(random_state)
    y_random = y.copy()
    y_random = np.random.permutation(y_random)  # Shuffle class labels

    return evaluate_image_classifier(
        images, y_random, 
        params=params, 
        fpr_budget=fpr_budget, 
        n_splits=n_splits, 
        reduce_dim=reduce_dim
        )


def bag_of_visual_words_basic(X, y, dataset_name, fpr_budget, plot_roc=False, hypersearch=False):

    images = [np.array(image.convert("RGB")) for image in X]

    # Hyperparameter grid
    param_grid = {
        'sift_clusters': [50, 70, 120],
        'model_type': ['random', 'gradient', 'stack', 'gaussian'],
        'n_components': [100, 175, 250]
    }

    # Step 1: Find best params
    if hypersearch:
        best_params, best_auc = hyperparam_search(images, y, param_grid, fpr_budget=fpr_budget, n_splits=5)
        print(f"Best AUC: {best_auc:.3f} | Params: {best_params}")
    else:
        best_params = {"sift_clusters": 70, "model_type": "stack", "n_components": 250}

    # Step 2: Evaluate on original task
    auc_scores, tpr_scores = evaluate_image_classifier(images, y, best_params, 
        fpr_budget=fpr_budget, n_splits=5, reduce_dim=True)
    print(f"Original Task (AUC): {mean(auc_scores):.4f} ± {pstdev(auc_scores):.4f}")
    print(f"Original Task (TPR@{fpr_budget}%FPR): {mean(tpr_scores):.4f} ± {pstdev(tpr_scores):.4f}")

    # Evaluate on random-split task (negative class only)
    rand_auc, rand_tpr = evaluate_random_labels(images, y, best_params, 
        fpr_budget=fpr_budget, n_splits=5, reduce_dim=True)
    print(f"Random-Split Task (AUC): {mean(rand_auc):.4f} ± {pstdev(rand_auc):.4f}")
    print(f"Random-Split Task (TPR@{fpr_budget}%FPR): {mean(rand_tpr):.4f} ± {pstdev(rand_tpr):.4f}")
