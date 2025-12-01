import operator
import numpy as np
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from statistics import mean, pstdev
from utils import *  

def extract_audio_features(audio_path, sr=22050, n_mfcc=20, ):
    """Extract all audio features and return as a flattened vector."""
    y, sr = librosa.load(audio_path, sr=sr)
    
    # --- Temporal Features ---
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()
    
    # --- Spectral Features ---
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    
    # --- Harmonic/Percussive ---
    harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr).mean(axis=1)
    
    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    
    # --- Tonnetz ---
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr).mean(axis=1)
    
    # --- Tempogram ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr).mean(axis=1)
    
    # Flatten all features into a single vector
    features = np.concatenate([
        [zcr, rms],
        [spectral_centroid, spectral_bandwidth, spectral_rolloff],
        chroma,
        mfccs,
        tonnetz,
        tempogram
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

def evaluate_audio_classifier(audio_paths, y, params, fpr_budget, n_splits=5, reduce_dim=True):
    """Train/evaluate a classifier on combined image features with CV."""

    X = np.array([
        extract_audio_features(path) for path in audio_paths
    ])
        
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


def hyperparam_search(audio_paths, y, param_grid, fpr_budget, n_splits=5):
    """Grid search over feature extraction and model hyperparameters."""
    best_auc = -1
    best_params = None
    
    for n_mfcc in param_grid.get('sift_clusters', [20]):
        for model_type in param_grid.get('model_type', ['random']):
            for n_components in param_grid.get('n_components', [100]):
                params = {
                'n_mfcc': n_mfcc,
                'model_type': model_type,
                'n_components': n_components,
                }
            auc, fpr = evaluate_audio_classifier(audio_paths, y, params, fpr_budget=fpr_budget, n_splits=n_splits)
            if (mean_auc := mean(auc)) > best_auc:
                best_auc = mean_auc
                best_params = params
    return best_params, best_auc


def evaluate_random_labels(audio_paths, y, params, fpr_budget, n_splits=5, random_state=42, reduce_dim=True):
    """Evaluate on randomly shuffled labels (negative class only)."""
    
    np.random.seed(random_state)
    y_random = y.copy()
    y_random = np.random.permutation(y_random)  # Shuffle class labels

    return evaluate_audio_classifier(
        audio_paths, y_random, 
        params=params, 
        fpr_budget=fpr_budget, 
        n_splits=n_splits, 
        reduce_dim=reduce_dim
        )


def bag_of_audio_words_basic(audio_paths, y, dataset_name, fpr_budget, plot_roc=False, hypersearch=False):

    # Hyperparameter grid
    param_grid = {
        'n_mfcc': [20, 35, 50],
        'model_type': ['random', 'gradient', 'stack', 'gaussian'],
        'n_components': [100, 175, 250]
    }

    # Step 1: Find best params
    if hypersearch:
        best_params, best_auc = hyperparam_search(audio_paths, y, param_grid, fpr_budget=fpr_budget, n_splits=5)
        print(f"Best AUC: {best_auc:.3f} | Params: {best_params}")
    else:
        best_params = {"n_mfcc": 20, "model_type": "stack", "n_components": 250}

    # Step 2: Evaluate on original task
    auc_scores, tpr_scores = evaluate_audio_classifier(audio_paths, y, best_params, 
        fpr_budget=fpr_budget, n_splits=5, reduce_dim=True)
    print(f"Original Task (AUC): {mean(auc_scores):.4f} ± {pstdev(auc_scores):.4f}")
    print(f"Original Task (TPR@{fpr_budget}%FPR): {mean(tpr_scores):.4f} ± {pstdev(tpr_scores):.4f}")

    # Evaluate on random-split task (negative class only)
    rand_auc, rand_tpr = evaluate_random_labels(audio_paths, y, best_params, 
        fpr_budget=fpr_budget, n_splits=5, reduce_dim=True)
    print(f"Random-Split Task (AUC): {mean(rand_auc):.4f} ± {pstdev(rand_auc):.4f}")
    print(f"Random-Split Task (TPR@{fpr_budget}%FPR): {mean(rand_tpr):.4f} ± {pstdev(rand_tpr):.4f}")
