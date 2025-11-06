#!/usr/bin/env python
"""拉取快乐8历史数据、训练多种模型并输出最新一期预测。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kl8 import (
    FeatureConfig,
    build_inference_features,
    build_training_matrix,
    ensemble_predict,
    fetch_draw_history,
    train_models,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-count", type=int, default=500, help="拉取最近多少期数据用于训练")
    parser.add_argument("--warmup", type=int, default=60, help="训练前跳过的期数，用于累积统计特征")
    parser.add_argument("--output", type=Path, default=None, help="可选，保存预测结果的 JSON 文件路径")
    parser.add_argument(
        "--min-top-prob",
        type=float,
        default=0.8,
        help="保证前若干个号码的最小概率阈值",
    )
    parser.add_argument("--top-n", type=int, default=3, help="需要强制提升的号码个数")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = FeatureConfig(warmup_draws=args.warmup)

    print("[1/5] 正在抓取快乐8历史数据……")
    history_df = fetch_draw_history(issue_count=args.issue_count)
    print(f"获取到 {len(history_df)} 期数据。最新一期：{history_df.iloc[-1]['issue']}")

    print("[2/5] 构建训练特征矩阵……")
    training_df = build_training_matrix(history_df, cfg=cfg)
    print(f"训练样本量：{len(training_df)}，特征维度：{len(cfg.feature_names)}")

    print("[3/5] 训练模型（逻辑回归、GBDT、随机森林）……")
    models = train_models(training_df, cfg=cfg)
    print(f"已训练 {len(models)} 个模型：{', '.join(models.keys())}")

    print("[4/5] 构建最新一期预测特征……")
    inference_df = build_inference_features(history_df, cfg=cfg)

    print("[5/5] 集成预测并输出结果……")
    result_df = ensemble_predict(
        models,
        inference_df,
        cfg=cfg,
        min_top_probability=args.min_top_prob,
        top_n=args.top_n,
    )

    pd.set_option("display.max_rows", 100)
    print(result_df.head(20))

    if args.output:
        payload: Dict[str, Any] = {
            "issue": str(history_df.iloc[-1]["issue"]),
            "predictions": result_df.to_dict(orient="records"),
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"预测结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
