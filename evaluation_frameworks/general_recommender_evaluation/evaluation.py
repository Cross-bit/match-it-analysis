#!/bin/python3
"""
Compatibility shim.

Historically, `SurpriseRatingBasedEvaluation` lived in this module.
The real implementation now lives in:
`evaluation_frameworks.general_recommender_evaluation.evaluation.surprise_rating_eval`.

This file only re-exports the class so legacy imports keep working:
`from evaluation_frameworks.general_recommender_evaluation.evaluation import SurpriseRatingBasedEvaluation`
"""

from evaluation_frameworks.general_recommender_evaluation.evaluation.surprise_rating_eval import (
    SurpriseRatingBasedEvaluation,
)

__all__ = ["SurpriseRatingBasedEvaluation"]
