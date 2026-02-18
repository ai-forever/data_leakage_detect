"""
Shift Detection: Baseline attacks for multimodal membership inference evaluation.
"""

from shift_detection.bag_of_words import bag_of_words_basic
from shift_detection.bag_of_visual_feats import bag_of_visual_words_basic
from shift_detection.bag_of_audio_feats import bag_of_audio_words_basic
from shift_detection.date_detection import (
    date_detection_arxiv,
    date_detection_basic,
)
from shift_detection.greedy_selection import (
    greedy_selection_arxiv,
    greedy_selection_basic,
    greedy_selection_wiki,
)
from shift_detection.utils import (
    get_dataset,
    get_roc_auc,
    get_tpr_metric,
    plot_tpr_fpr_curve,
)

__all__ = [
    # Attack functions
    "bag_of_words_basic",
    "bag_of_visual_words_basic",
    "bag_of_audio_words_basic",
    "date_detection_arxiv",
    "date_detection_basic",
    "greedy_selection_arxiv",
    "greedy_selection_basic",
    "greedy_selection_wiki",
    # Utility functions
    "get_dataset",
    "get_roc_auc",
    "get_tpr_metric",
    "plot_tpr_fpr_curve",
]
