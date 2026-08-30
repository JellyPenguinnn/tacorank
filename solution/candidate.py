"""Production candidate entrypoint with a deterministic popularity baseline."""

from __future__ import annotations

import csv
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def run(invocation: Any) -> None:
    """Fit on the mounted train view and score the requested ordered population."""

    train_path = _regular_file(invocation.input_root / "train.csv")
    score_path = _regular_file(invocation.input_root / "score.csv")
    positives: Counter[str] = Counter()
    impressions: Counter[str] = Counter()
    total_positive = 0
    total_impressions = 0
    with train_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if not {"video_id", "long_view"}.issubset(reader.fieldnames or ()):
            raise ValueError("train.csv is missing required columns")
        for row in reader:
            label = int(row["long_view"])
            if label not in (0, 1):
                raise ValueError("train.csv long_view must be binary")
            video_id = row["video_id"]
            impressions[video_id] += 1
            positives[video_id] += label
            total_positive += label
            total_impressions += 1
    if total_impressions == 0:
        raise ValueError("train.csv is empty")
    global_mean = total_positive / total_impressions
    prior = 20.0
    scores: Dict[str, float] = {
        video_id: (positives[video_id] + prior * global_mean)
        / (count + prior)
        for video_id, count in impressions.items()
    }

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(invocation.output_path, flags, 0o600)
    with score_path.open(newline="", encoding="utf-8") as source, os.fdopen(
        descriptor, "w", newline="", encoding="utf-8"
    ) as output:
        reader = csv.DictReader(source, strict=True)
        required = {"row_id", "user_id", "video_id"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("score.csv is missing required columns")
        writer = csv.writer(output)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for expected, row in enumerate(reader):
            if int(row["row_id"]) != expected:
                raise ValueError("score.csv row_id must be contiguous and ordered")
            score = scores.get(row["video_id"], global_mean)
            if not math.isfinite(score):
                raise ValueError("candidate produced a non-finite score")
            writer.writerow((expected, row["user_id"], row["video_id"], format(score, ".12g")))
        output.flush()
        os.fsync(output.fileno())


def _regular_file(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("required candidate input is missing")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ValueError("candidate inputs must use canonical paths")
    return resolved
