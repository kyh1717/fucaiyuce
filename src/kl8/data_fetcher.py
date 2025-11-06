"""从中国福利彩票官网抓取快乐8往期开奖数据。"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import requests

__all__ = ["DrawRecord", "fetch_draw_history"]

_API_URL = "https://www.cwl.gov.cn/cwl_admin/kjxx/findDrawNotice"


@dataclass
class DrawRecord:
    """表示单期开奖的数据。"""

    issue: str
    draw_date: _dt.date
    numbers: Sequence[int]

    @property
    def sum_value(self) -> int:
        return int(sum(self.numbers))

    @property
    def average(self) -> float:
        return float(self.sum_value) / len(self.numbers)


def _parse_numbers(raw: str) -> List[int]:
    cleaned = raw.replace("，", ",").replace(" ", ",").replace("-", ",")
    parts = [p for p in cleaned.replace("..", ".").replace("..", ".").split(",") if p]
    return [int(p) for p in parts]


def _normalise_date(raw: str) -> _dt.date:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return _dt.datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期格式: {raw}")


def _request_history(
    issue_count: int,
    start_issue: Optional[str] = None,
    end_issue: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> List[dict]:
    sess = session or requests.Session()
    params = {"name": "kl8", "issueCount": issue_count}
    if start_issue:
        params["issueStart"] = start_issue
    if end_issue:
        params["issueEnd"] = end_issue

    response = sess.get(_API_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not payload or "result" not in payload:
        raise RuntimeError("未能获取到快乐8历史数据。")
    return payload["result"]


def _records_from_payload(data: Iterable[dict]) -> List[DrawRecord]:
    records: List[DrawRecord] = []
    for item in data:
        code = item.get("code") or item.get("issue") or item.get("qh")
        date_str = item.get("date") or item.get("time") or item.get("dateTime")
        numbers_str = item.get("red") or item.get("result") or item.get("redContent")
        if not code or not date_str or not numbers_str:
            continue
        record = DrawRecord(
            issue=str(code),
            draw_date=_normalise_date(str(date_str)),
            numbers=tuple(sorted(_parse_numbers(str(numbers_str)))),
        )
        records.append(record)
    return records


def fetch_draw_history(
    issue_count: int = 300,
    start_issue: Optional[str] = None,
    end_issue: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """抓取快乐8往期数据并返回 DataFrame。"""

    raw_records = _request_history(
        issue_count=issue_count, start_issue=start_issue, end_issue=end_issue, session=session
    )
    records = _records_from_payload(raw_records)
    if not records:
        raise RuntimeError("未能解析快乐8历史记录。")

    df = pd.DataFrame(
        {
            "issue": [r.issue for r in records],
            "draw_date": [r.draw_date for r in records],
            "numbers": [list(r.numbers) for r in records],
            "sum": [r.sum_value for r in records],
            "average": [r.average for r in records],
        }
    )
    df = df.sort_values("issue").reset_index(drop=True)
    return df
