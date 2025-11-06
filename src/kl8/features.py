"""特征工程：将快乐8往期记录转换为建模所需的特征矩阵。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["FeatureConfig", "build_training_matrix", "build_inference_features"]


@dataclass(frozen=True)
class FeatureConfig:
    """特征配置。"""

    warmup_draws: int = 30
    window_sizes: Tuple[int, ...] = (10, 30, 50)

    @property
    def feature_names(self) -> List[str]:
        feats = ["overall_freq", "appearance_rate"]
        feats.extend([f"win_rate_{w}" for w in self.window_sizes])
        feats.extend([f"gap_ratio_{w}" for w in self.window_sizes])
        feats.append("recent_gap")
        return feats


def _initialise_state(num_balls: int) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    counts = np.zeros(num_balls, dtype=float)
    last_seen = np.full(num_balls, -1, dtype=int)
    history: List[np.ndarray] = []
    return counts, last_seen, history


def _update_state(
    counts: np.ndarray,
    last_seen: np.ndarray,
    history: List[np.ndarray],
    mask: np.ndarray,
    draw_index: int,
) -> None:
    counts += mask
    last_seen[mask == 1] = draw_index
    history.append(mask.copy())


def _window_stats(history: List[np.ndarray], window: int, num_balls: int) -> np.ndarray:
    if not history:
        return np.zeros(num_balls, dtype=float)
    window_history = history[-window:]
    return np.sum(window_history, axis=0) / float(len(window_history))


def _gap_ratios(last_seen: np.ndarray, draw_index: int, window: int) -> np.ndarray:
    gaps = np.where(last_seen >= 0, draw_index - last_seen, draw_index + window)
    return np.clip(gaps / float(window), 0.0, 2.0)


def _build_feature_row(
    number: int,
    overall_freq: float,
    appearance_rate: float,
    win_rates: Sequence[float],
    gap_ratios: Sequence[float],
    recent_gap: float,
) -> dict:
    row = {
        "number": number,
        "overall_freq": overall_freq,
        "appearance_rate": appearance_rate,
        "recent_gap": recent_gap,
    }
    for idx, value in enumerate(win_rates):
        row[f"win_rate_{idx}"] = value
    for idx, value in enumerate(gap_ratios):
        row[f"gap_ratio_{idx}"] = value
    return row


def _ensure_column_names(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    rename_map = {}
    for idx, window in enumerate(cfg.window_sizes):
        rename_map[f"win_rate_{idx}"] = f"win_rate_{window}"
        rename_map[f"gap_ratio_{idx}"] = f"gap_ratio_{window}"
    return df.rename(columns=rename_map)


def build_training_matrix(draw_history: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    cfg = cfg or FeatureConfig()
    numbers_per_draw = 80

    draw_history = draw_history.sort_values("issue").reset_index(drop=True)
    masks = []
    for _, row in draw_history.iterrows():
        mask = np.zeros(numbers_per_draw, dtype=float)
        mask[np.array(row["numbers"], dtype=int) - 1] = 1.0
        masks.append(mask)

    counts, last_seen, history = _initialise_state(numbers_per_draw)
    rows = []
    labels = []

    for draw_idx, mask in enumerate(masks):
        if draw_idx >= cfg.warmup_draws:
            total_draws = float(draw_idx)
            overall_freq = counts / max(total_draws, 1.0)
            appearance_rate = counts / max(np.sum(counts), 1.0)
            win_rates = [_window_stats(history, w, numbers_per_draw) for w in cfg.window_sizes]
            gap_ratios = [_gap_ratios(last_seen, draw_idx, w) for w in cfg.window_sizes]
            recent_gap = np.where(last_seen >= 0, draw_idx - last_seen, draw_idx)

            for num in range(numbers_per_draw):
                row = _build_feature_row(
                    number=num + 1,
                    overall_freq=overall_freq[num],
                    appearance_rate=appearance_rate[num],
                    win_rates=[win_rates_i[num] for win_rates_i in win_rates],
                    gap_ratios=[gap_ratios_i[num] for gap_ratios_i in gap_ratios],
                    recent_gap=recent_gap[num],
                )
                rows.append(row)
                labels.append(mask[num])

        _update_state(counts, last_seen, history, mask, draw_idx)

    if not rows:
        raise ValueError("样本数量不足，无法构建训练矩阵。")

    df = pd.DataFrame(rows)
    df["label"] = labels
    df = _ensure_column_names(df, cfg)
    return df


def build_inference_features(draw_history: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    cfg = cfg or FeatureConfig()
    numbers_per_draw = 80

    draw_history = draw_history.sort_values("issue").reset_index(drop=True)
    masks = []
    for _, row in draw_history.iterrows():
        mask = np.zeros(numbers_per_draw, dtype=float)
        mask[np.array(row["numbers"], dtype=int) - 1] = 1.0
        masks.append(mask)

    counts, last_seen, history = _initialise_state(numbers_per_draw)

    for draw_idx, mask in enumerate(masks):
        _update_state(counts, last_seen, history, mask, draw_idx)

    total_draws = float(len(masks))
    overall_freq = counts / max(total_draws, 1.0)
    appearance_rate = counts / max(np.sum(counts), 1.0)
    win_rates = [_window_stats(history, w, numbers_per_draw) for w in cfg.window_sizes]
    gap_ratios = [_gap_ratios(last_seen, len(masks), w) for w in cfg.window_sizes]
    recent_gap = np.where(last_seen >= 0, len(masks) - last_seen, len(masks))

    rows = []
    for num in range(numbers_per_draw):
        row = _build_feature_row(
            number=num + 1,
            overall_freq=overall_freq[num],
            appearance_rate=appearance_rate[num],
            win_rates=[win_rates_i[num] for win_rates_i in win_rates],
            gap_ratios=[gap_ratios_i[num] for gap_ratios_i in gap_ratios],
            recent_gap=recent_gap[num],
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _ensure_column_names(df, cfg)
    return df
