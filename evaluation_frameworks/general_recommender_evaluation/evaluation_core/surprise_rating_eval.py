#!/bin/python3
"""
Thin compatibility wrapper.

The canonical implementation of `SurpriseRatingBasedEvaluation` now lives in
`evaluation_frameworks.general_recommender_evaluation.evaluation`.
"""

from evaluation_frameworks.general_recommender_evaluation.evaluation import (
    SurpriseRatingBasedEvaluation,
)

__all__ = ["SurpriseRatingBasedEvaluation"]
