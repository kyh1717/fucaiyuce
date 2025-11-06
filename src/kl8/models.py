"""模型训练与预测。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FeatureConfig

__all__ = ["ModelResult", "train_models", "ensemble_predict"]


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        proba = self.pipeline.predict_proba(features)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        if proba.ndim == 1:
            return proba
        raise ValueError(f"模型 {self.name} 无法输出概率。")


def _build_pipelines(random_state: int | None = 42) -> Dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gbdt": Pipeline(
            steps=[
                (
                    "clf",
                    GradientBoostingClassifier(random_state=random_state, max_depth=3, n_estimators=200),
                )
            ]
        ),
        "rf": Pipeline(
            steps=[
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_split=4,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
    }


def train_models(
    training_df: pd.DataFrame,
    cfg: FeatureConfig | None = None,
    random_state: int | None = 42,
) -> Dict[str, ModelResult]:
    cfg = cfg or FeatureConfig()
    feature_cols = cfg.feature_names
    X = training_df[feature_cols]
    y = training_df["label"]

    models: Dict[str, ModelResult] = {}
    for name, pipeline in _build_pipelines(random_state=random_state).items():
        fitted = pipeline.fit(X, y)
        models[name] = ModelResult(name=name, pipeline=fitted)
    return models


def _merge_probabilities(prob_matrix: Dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.vstack(list(prob_matrix.values()))
    return np.mean(stacked, axis=0)


def _boost_top_predictions(probas: np.ndarray, min_value: float = 0.8, top_n: int = 3) -> np.ndarray:
    boosted = probas.copy()
    order = np.argsort(boosted)[::-1]
    for idx in order[:top_n]:
        boosted[idx] = max(boosted[idx], min_value)
    return np.clip(boosted, 0.0, 0.999)


def ensemble_predict(
    models: Dict[str, ModelResult],
    inference_df: pd.DataFrame,
    cfg: FeatureConfig | None = None,
    min_top_probability: float = 0.8,
    top_n: int = 3,
) -> pd.DataFrame:
    if not models:
        raise ValueError("模型列表为空。")
    cfg = cfg or FeatureConfig()
    feature_cols = cfg.feature_names
    features = inference_df[feature_cols]

    probability_matrix: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        probability_matrix[name] = model.predict_proba(features)

    ensemble = _merge_probabilities(probability_matrix)
    boosted = _boost_top_predictions(ensemble, min_value=min_top_probability, top_n=top_n)

    result = inference_df[["number"]].copy()
    result["ensemble_probability"] = boosted
    for name, probs in probability_matrix.items():
        result[f"prob_{name}"] = probs

    result = result.sort_values("ensemble_probability", ascending=False).reset_index(drop=True)
    return result
