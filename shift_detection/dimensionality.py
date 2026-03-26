"""Safe PCA + scaling for small sample sizes (n_components <= min(n_samples, n_features))."""

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def reduce_dimensionality(features, n_components=100):
    """StandardScaler + PCA; caps components so PCA never exceeds min(n_samples, n_features)."""
    n_samples, n_features = features.shape
    max_pca = min(n_samples, n_features)
    n_components = max(1, min(int(n_components), n_features, max_pca))
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    scaler = StandardScaler()
    pipeline = Pipeline([("scaler", scaler), ("pca", pca)])
    reduced_features = pipeline.fit_transform(features)
    return reduced_features, pipeline
