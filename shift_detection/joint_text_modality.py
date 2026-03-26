"""
Joint shift evaluation: concatenate text statistics (TF-IDF / count) with
modality feature vectors (image BoVW-style stats or audio spectral stats).
"""

from statistics import mean, pstdev

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shift_detection.bag_of_audio_feats import extract_audio_features, get_model as get_audio_model
from shift_detection.bag_of_visual_feats import extract_all_features, get_model as get_visual_model
from shift_detection.dimensionality import reduce_dimensionality
from shift_detection.utils import get_roc_auc, get_tpr_metric, stratified_cv_split_indices


def _vectorize_text_train_test(
    texts_train,
    texts_test,
    max_features: int,
    vectorizer: str,
):
    if vectorizer == "tf":
        vec = TfidfVectorizer(max_features=max_features)
    else:
        vec = CountVectorizer(max_features=max_features)
    Xtr = vec.fit_transform(texts_train).toarray()
    Xte = vec.transform(texts_test).toarray()
    return Xtr, Xte


def _modality_feature_matrix(paths: list, modality_type: str, params: dict) -> np.ndarray:
    if modality_type == "image":
        from PIL import Image

        images = [np.array(Image.open(p).convert("RGB")) for p in paths]
        return extract_all_features(
            images,
            sift_clusters=params.get("sift_clusters", 50),
            dct_coeffs=params.get("dct_coeffs", 20),
            color_bins=params.get("color_bins", 32),
            lbp_points=params.get("lbp_points", 8),
        )
    if modality_type == "audio":
        return np.array(
            [extract_audio_features(p, n_mfcc=params.get("n_mfcc", 20)) for p in paths]
        )
    raise ValueError(f"Unknown modality_type: {modality_type!r} (use 'image' or 'audio')")


def evaluate_joint_classifier(
    texts: list[str],
    modality_paths: list,
    y,
    params: dict,
    fpr_budget: float,
    n_splits: int = 5,
    reduce_dim: bool = True,
):
    """
    Per fold: fit text vectorizer on train only; hstack with precomputed modality features;
    scale on train; optional PCA (train-only fit); classify.
    """
    y_arr = np.asarray(y)
    texts_arr = np.asarray(texts, dtype=object)
    modality_type = params["modality_type"]
    X_mod = _modality_feature_matrix(modality_paths, modality_type, params)
    splits = stratified_cv_split_indices(y_arr, n_splits=n_splits, random_state=42)
    get_model = (
        get_visual_model if modality_type == "image" else get_audio_model
    )
    model_type = params.get("model_type", "stack")

    roc_auc_scores, tpr_scores = [], []
    for train_idx, test_idx in splits:
        texts_tr = texts_arr[train_idx].tolist()
        texts_te = texts_arr[test_idx].tolist()

        Xt_tr, Xt_te = _vectorize_text_train_test(
            texts_tr,
            texts_te,
            max_features=params.get("max_text_features", 62),
            vectorizer=params.get("text_vectorizer", "tf"),
        )
        Xm_tr, Xm_te = X_mod[train_idx], X_mod[test_idx]
        Xtr = np.hstack([Xt_tr, Xm_tr])
        Xte = np.hstack([Xt_te, Xm_te])

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        if reduce_dim:
            n_comp = params.get("n_components", 100)
            Xtr, pipe = reduce_dimensionality(Xtr, n_components=n_comp)
            Xte = pipe.transform(Xte)

        model = get_model(model_type)
        model.fit(Xtr, y_arr[train_idx])
        y_pred_proba = model.predict_proba(Xte)[:, 1]

        y_te = y_arr[test_idx]
        try:
            roc_auc_scores.append(get_roc_auc(y_te, y_pred_proba))
            tpr_scores.append(get_tpr_metric(y_te, y_pred_proba, fpr_budget))
        except ValueError:
            roc_auc_scores.append(0.5)
            tpr_scores.append(fpr_budget / 100.0)

    return roc_auc_scores, tpr_scores


def evaluate_random_labels_joint(
    texts,
    modality_paths,
    y,
    params,
    fpr_budget,
    n_splits=5,
    random_state=42,
    reduce_dim=True,
):
    y_random = np.random.RandomState(random_state).permutation(np.asarray(y))
    return evaluate_joint_classifier(
        texts,
        modality_paths,
        y_random,
        params,
        fpr_budget=fpr_budget,
        n_splits=n_splits,
        reduce_dim=reduce_dim,
    )


def joint_text_modality_basic(X, y, dataset_name, fpr_budget, plot_roc=False, hypersearch=False):
    texts = X["texts"]
    modality_paths = X["modality_paths"]
    modality_type = X["modality_type"]

    params = {
        "modality_type": modality_type,
        "model_type": "stack",
        "max_text_features": 62,
        "text_vectorizer": "tf",
        "n_components": 100,
        "sift_clusters": 50,
        "dct_coeffs": 20,
        "color_bins": 32,
        "lbp_points": 8,
        "n_mfcc": 20,
    }
    if hypersearch:
        best_auc = -1.0
        best_params = params
        for max_tf in (30, 62):
            for tv in ("tf", "count"):
                for n_comp in (50, 100, 175):
                    trial = {
                        **params,
                        "max_text_features": max_tf,
                        "text_vectorizer": tv,
                        "n_components": n_comp,
                    }
                    aucs, _ = evaluate_joint_classifier(
                        texts,
                        modality_paths,
                        y,
                        trial,
                        fpr_budget=fpr_budget,
                        n_splits=5,
                    )
                    m = mean(aucs)
                    if m > best_auc:
                        best_auc = m
                        best_params = trial
        params = best_params
        print(f"Joint hypersearch best mean AUC: {best_auc:.4f} | Params: {params}")

    auc_scores, tpr_scores = evaluate_joint_classifier(
        texts, modality_paths, y, params, fpr_budget=fpr_budget, n_splits=5
    )
    print(f"Joint Original Task (AUC): {mean(auc_scores):.4f} ± {pstdev(auc_scores):.4f}")
    print(
        f"Joint Original Task (TPR@{fpr_budget}%FPR): {mean(tpr_scores):.4f} ± {pstdev(tpr_scores):.4f}"
    )

    rand_auc, rand_tpr = evaluate_random_labels_joint(
        texts, modality_paths, y, params, fpr_budget=fpr_budget, n_splits=5
    )
    print(f"Joint Random-Split Task (AUC): {mean(rand_auc):.4f} ± {pstdev(rand_auc):.4f}")
    print(
        f"Joint Random-Split Task (TPR@{fpr_budget}%FPR): {mean(rand_tpr):.4f} ± {pstdev(rand_tpr):.4f}"
    )

    if plot_roc:
        y_arr = np.asarray(y)
        idx = np.arange(len(y_arr))
        try:
            tr, te = train_test_split(
                idx,
                test_size=0.25,
                stratify=y_arr,
                random_state=42,
            )
        except ValueError:
            tr, te = train_test_split(idx, test_size=0.25, random_state=42)
        texts_arr = np.asarray(texts, dtype=object)
        texts_tr = texts_arr[tr].tolist()
        texts_te = texts_arr[te].tolist()
        Xt_tr, Xt_te = _vectorize_text_train_test(
            texts_tr,
            texts_te,
            max_features=params["max_text_features"],
            vectorizer=params["text_vectorizer"],
        )
        X_mod = _modality_feature_matrix(modality_paths, modality_type, params)
        Xm_tr, Xm_te = X_mod[tr], X_mod[te]
        Xtr = np.hstack([Xt_tr, Xm_tr])
        Xte = np.hstack([Xt_te, Xm_te])
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        Xtr, pipe = reduce_dimensionality(Xtr, n_components=params["n_components"])
        Xte = pipe.transform(Xte)
        get_model = get_visual_model if modality_type == "image" else get_audio_model
        model = get_model(params["model_type"])
        model.fit(Xtr, y_arr[tr])
        y_pred_proba = model.predict_proba(Xte)[:, 1]
        from shift_detection.utils import plot_tpr_fpr_curve

        plot_tpr_fpr_curve(y_arr[te], y_pred_proba, fpr_budget)
