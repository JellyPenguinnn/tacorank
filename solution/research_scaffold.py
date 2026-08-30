"""Standard-library trainable FM scaffold for TacoRank experiments."""

from __future__ import annotations

from array import array
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import zlib


FEATURES = ("user_id", "video_id", "author_id", "tab", "duration_ms")
FORMULATIONS = {"passthrough", "pointwise", "bpr", "listwise", "temporal_history"}
HASH_DIMENSION = 1 << 16
PredictionRow = Tuple[int, str, str, float]
TrainingRow = Tuple[Tuple[str, ...], str, int, float]


def run_experiment(invocation: Any, raw_config: Mapping[str, Any]) -> None:
    config = validate_config(raw_config)
    score_path, parent_path = _validated_inputs(invocation.input_root)
    formulation = str(config["formulation"])
    if formulation == "passthrough":
        _validate_alignment(score_path, parent_path)
        _exclusive_copy(parent_path, invocation.output_path)
        _write_diagnostics(
            invocation.output_path,
            _empty_diagnostics(formulation, getattr(invocation, "seed", 0)),
        )
        return

    train = _load_training(
        _regular_file(invocation.input_root / "train.csv"),
        str(invocation.fidelity),
        config,
    )
    model, diagnostics = _fit(train, config, int(invocation.seed or 0))
    prediction_stats = _write_model_predictions(
        invocation.output_path, score_path, parent_path, model, config
    )
    diagnostics.update(
        formulation=formulation,
        seed=int(invocation.seed or 0),
        train_rows=len(train),
        **prediction_stats,
    )
    _write_diagnostics(invocation.output_path, diagnostics)


def validate_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    bounds = {
        "embedding_dim": (2, 32, int),
        "learning_rate": (1e-5, 0.2, float),
        "epochs": (1, 8, int),
        "negative_count": (1, 16, int),
        "l2": (0.0, 0.1, float),
        "residual_scale": (0.0, 0.5, float),
        "max_train_rows": (1000, 250_000, int),
        "history_decay_days": (1.0, 180.0, float),
        "history_shrinkage": (0.0, 1000.0, float),
    }
    unknown = set(raw) - {"family", "formulation", *bounds}
    if unknown:
        raise ValueError("unknown experiment configuration: %s" % sorted(unknown))
    config = dict(raw)
    if str(config.get("formulation", "")) not in FORMULATIONS:
        raise ValueError("unsupported formulation")
    for key, (lower, upper, kind) in bounds.items():
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("%s must be numeric" % key)
        if kind is int and not isinstance(value, int):
            raise ValueError("%s must be an integer" % key)
        numeric = float(value)
        if not math.isfinite(numeric) or not lower <= numeric <= upper:
            raise ValueError("%s must be in [%g, %g]" % (key, lower, upper))
    return config


def _load_training(
    path: Path, fidelity: str, config: Mapping[str, Any]
) -> List[TrainingRow]:
    limits = {
        "smoke": 20_000,
        "proxy": 100_000,
        "full": int(config["max_train_rows"]),
        "final": int(config["max_train_rows"]),
    }
    limit = min(limits.get(fidelity, limits["full"]), int(config["max_train_rows"]))
    rows: List[Tuple[int, TrainingRow]] = []
    representative = fidelity in {"full", "final"}
    rng = random.Random(0x5A17)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = set(FEATURES) | {"date", "long_view"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("train.csv is missing required columns")
        for row_index, row in enumerate(reader):
            label = float(row["long_view"])
            if label not in (0.0, 1.0):
                raise ValueError("training labels must be binary")
            if representative and len(rows) >= limit:
                replacement = rng.randrange(row_index + 1)
                if replacement >= limit:
                    continue
            parsed = (
                _feature_tokens(row),
                str(row["user_id"]),
                _date_ordinal(row["date"]),
                label,
            )
            if representative and len(rows) >= limit:
                rows[replacement] = (row_index, parsed)
            else:
                rows.append((row_index, parsed))
            if not representative and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("train.csv is empty")
    return [row for _, row in sorted(rows)]


def _fit(
    train: Sequence[TrainingRow], config: Mapping[str, Any], seed: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if config["formulation"] == "temporal_history":
        return _fit_temporal(train, config), {
            "loss_curve": [0.0],
            "loss_start": 0.0,
            "loss_end": 0.0,
            "pairwise_accuracy": 0.0,
            "gradient_norm": 0.0,
        }

    rng = random.Random(seed)
    rank = int(config["embedding_dim"])
    weights = array("d", [0.0]) * HASH_DIMENSION
    factors = array(
        "d",
        (rng.gauss(0.0, 0.01) for _ in range(HASH_DIMENSION * rank)),
    )
    indices = [_hashed_indices(row[0]) for row in train]
    labels = [row[3] for row in train]
    seen_tokens = {token for row in train for token in row[0]}
    losses: List[float] = []
    gradient_norms: List[float] = []
    ranking_accuracy = 0.0
    learning_rate = float(config["learning_rate"])
    l2 = float(config["l2"])

    for _ in range(int(config["epochs"])):
        loss_total = 0.0
        norm_total = 0.0
        updates = 0
        if config["formulation"] == "pointwise":
            order = list(range(len(train)))
            rng.shuffle(order)
            for row_index in order:
                score = _fm_score(weights, factors, indices[row_index], rank)
                probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
                coefficient = probability - labels[row_index]
                loss_total += _softplus(score) - labels[row_index] * score
                norm_total += _apply_group_update(
                    weights,
                    factors,
                    (indices[row_index],),
                    (coefficient,),
                    rank,
                    learning_rate,
                    l2,
                )
                updates += 1
        elif config["formulation"] == "bpr":
            pairs = _pairwise_rows(train, rng)
            correct = 0
            for positive, negative in pairs:
                difference = _fm_score(
                    weights, factors, indices[positive], rank
                ) - _fm_score(weights, factors, indices[negative], rank)
                coefficient = -1.0 / (
                    1.0 + math.exp(max(-30.0, min(30.0, difference)))
                )
                loss_total += _softplus(-difference)
                correct += int(difference > 0.0)
                norm_total += _apply_group_update(
                    weights,
                    factors,
                    (indices[positive], indices[negative]),
                    (coefficient, -coefficient),
                    rank,
                    learning_rate,
                    l2,
                )
                updates += 1
            ranking_accuracy = correct / len(pairs)
        else:
            groups = _listwise_rows(
                train, rng, int(config["negative_count"])
            )
            correct = 0
            for group in groups:
                scores = [
                    _fm_score(weights, factors, indices[row_index], rank)
                    for row_index in group
                ]
                maximum = max(scores)
                exponentials = [math.exp(score - maximum) for score in scores]
                denominator = sum(exponentials)
                probabilities = [value / denominator for value in exponentials]
                coefficients = [probabilities[0] - 1.0, *probabilities[1:]]
                loss_total += math.log(denominator) - (scores[0] - maximum)
                correct += int(scores[0] > max(scores[1:]))
                norm_total += _apply_group_update(
                    weights,
                    factors,
                    tuple(indices[row_index] for row_index in group),
                    tuple(coefficients),
                    rank,
                    learning_rate,
                    l2,
                )
                updates += 1
            ranking_accuracy = correct / len(groups)
        if updates == 0:
            raise ValueError("objective produced no trainable updates")
        losses.append(loss_total / updates)
        gradient_norms.append(norm_total / updates)

    return {
        "kind": "fm",
        "weights": weights,
        "factors": factors,
        "rank": rank,
        "seen_tokens": seen_tokens,
    }, {
        "loss_curve": losses,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "pairwise_accuracy": ranking_accuracy,
        "gradient_norm": gradient_norms[-1],
    }


def _pairwise_rows(
    train: Sequence[TrainingRow], rng: random.Random
) -> List[Tuple[int, int]]:
    grouped = _group_label_rows(train)
    pairs = []
    for user in sorted(grouped):
        positive, negative = grouped[user]
        if not positive or not negative:
            continue
        pairs.extend((row, rng.choice(negative)) for row in positive)
    if not pairs:
        raise ValueError("BPR requires within-user positive-negative pairs")
    return pairs


def _listwise_rows(
    train: Sequence[TrainingRow], rng: random.Random, negative_count: int
) -> List[Tuple[int, ...]]:
    grouped = _group_label_rows(train)
    groups = []
    for user in sorted(grouped):
        positive, negative = grouped[user]
        if not positive or not negative:
            continue
        for row in positive:
            groups.append(
                (row, *(rng.choice(negative) for _ in range(negative_count)))
            )
    if not groups:
        raise ValueError("listwise objective requires informative within-user lists")
    return groups


def _group_label_rows(
    train: Sequence[TrainingRow],
) -> Dict[str, Tuple[List[int], List[int]]]:
    grouped: Dict[str, Tuple[List[int], List[int]]] = {}
    for index, row in enumerate(train):
        positive, negative = grouped.setdefault(row[1], ([], []))
        (positive if row[3] > 0.5 else negative).append(index)
    return grouped


def _fit_temporal(
    train: Sequence[TrainingRow], config: Mapping[str, Any]
) -> Dict[str, Any]:
    latest = max(row[2] for row in train)
    mean_label = sum(row[3] for row in train) / len(train)
    values: Dict[str, List[float]] = {}
    for tokens, _, date, label in train:
        weight = math.exp(
            -max(0, latest - date) / float(config["history_decay_days"])
        )
        for token in tokens[:3]:
            entry = values.setdefault(token, [0.0, 0.0])
            entry[0] += (label - mean_label) * weight
            entry[1] += weight
    return {"kind": "temporal", "values": values}


def _fm_score(
    weights: array, factors: array, indices: Sequence[int], rank: int
) -> float:
    score = sum(weights[index] for index in indices)
    for factor in range(rank):
        total = 0.0
        square_total = 0.0
        for index in indices:
            value = factors[index * rank + factor]
            total += value
            square_total += value * value
        score += 0.5 * (total * total - square_total)
    return score


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _apply_group_update(
    weights: array,
    factors: array,
    rows: Sequence[Sequence[int]],
    coefficients: Sequence[float],
    rank: int,
    learning_rate: float,
    l2: float,
) -> float:
    linear_gradients: Dict[int, float] = {}
    factor_gradients: Dict[int, float] = {}
    squared = 0.0
    count = 0
    for indices, coefficient in zip(rows, coefficients):
        sums = [
            sum(factors[index * rank + factor] for index in indices)
            for factor in range(rank)
        ]
        for index in indices:
            linear = coefficient + l2 * weights[index]
            linear_gradients[index] = linear_gradients.get(index, 0.0) + linear
            squared += linear * linear
            count += 1
            for factor in range(rank):
                offset = index * rank + factor
                old = factors[offset]
                gradient = coefficient * (sums[factor] - old) + l2 * old
                factor_gradients[offset] = (
                    factor_gradients.get(offset, 0.0) + gradient
                )
                squared += gradient * gradient
                count += 1
    for index, gradient in linear_gradients.items():
        weights[index] -= learning_rate * gradient
    for offset, gradient in factor_gradients.items():
        factors[offset] -= learning_rate * gradient
    return math.sqrt(squared / max(1, count))


def _write_model_predictions(
    output_path: Path,
    score_path: Path,
    parent_path: Path,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, float]:
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    residual_sum = 0.0
    residual_square_sum = 0.0
    covered = 0
    count = 0
    try:
        with score_path.open(newline="", encoding="utf-8") as score_handle, parent_path.open(
            newline="", encoding="utf-8"
        ) as parent_handle, os.fdopen(
            descriptor, "w", newline="", encoding="utf-8"
        ) as output:
            descriptor = -1
            score_rows = csv.DictReader(score_handle, strict=True)
            parent_rows = csv.DictReader(parent_handle, strict=True)
            _validate_headers(score_rows.fieldnames, parent_rows.fieldnames)
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(("row_id", "user_id", "video_id", "score"))
            sentinel = object()
            while True:
                score_row = next(score_rows, sentinel)
                parent_row = next(parent_rows, sentinel)
                if score_row is sentinel and parent_row is sentinel:
                    break
                if score_row is sentinel or parent_row is sentinel:
                    raise ValueError("frozen FM predictions have the wrong row count")
                _validate_row_alignment(count, score_row, parent_row)
                tokens = _feature_tokens(score_row)
                if model["kind"] == "fm":
                    raw = _fm_score(
                        model["weights"],
                        model["factors"],
                        _hashed_indices(tokens),
                        model["rank"],
                    )
                    is_covered = any(token in model["seen_tokens"] for token in tokens)
                else:
                    raw = 0.0
                    is_covered = False
                    for token in tokens[:3]:
                        entry = model["values"].get(token)
                        if entry is not None:
                            raw += entry[0] / (
                                entry[1] + float(config["history_shrinkage"])
                            )
                            is_covered = True
                residual = math.tanh(raw)
                value = float(parent_row["score"]) + float(
                    config["residual_scale"]
                ) * residual
                if not math.isfinite(value):
                    raise ValueError("candidate produced non-finite predictions")
                writer.writerow(
                    (
                        count,
                        score_row["user_id"],
                        score_row["video_id"],
                        "%.17g" % value,
                    )
                )
                residual_sum += residual
                residual_square_sum += residual * residual
                covered += int(is_covered)
                count += 1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if count == 0:
        raise ValueError("score population is empty")
    mean = residual_sum / count
    variance = max(0.0, residual_square_sum / count - mean * mean)
    return {
        "interaction_coverage": covered / count,
        "residual_mean": mean,
        "residual_std": math.sqrt(variance),
    }


def self_test() -> None:
    """Check determinism, user-bounded sampling, and the BPR gradient."""

    rows: List[TrainingRow] = [
        (("user_id=1", "video_id=1", "author_id=1", "tab=0", "duration_ms=1"), "1", 1, 1.0),
        (("user_id=1", "video_id=2", "author_id=2", "tab=0", "duration_ms=2"), "1", 2, 0.0),
        (("user_id=2", "video_id=3", "author_id=3", "tab=1", "duration_ms=3"), "2", 1, 1.0),
        (("user_id=2", "video_id=4", "author_id=4", "tab=1", "duration_ms=4"), "2", 2, 0.0),
    ]
    pairs = _pairwise_rows(rows, random.Random(7))
    if any(rows[positive][1] != rows[negative][1] for positive, negative in pairs):
        raise ValueError("negative sampling crossed user boundaries")
    groups = _listwise_rows(rows, random.Random(7), 2)
    if any(len({rows[index][1] for index in group}) != 1 for group in groups):
        raise ValueError("listwise sampling crossed user boundaries")
    rank = 2
    weights = array("d", [0.0]) * HASH_DIMENSION
    factors = array("d", [0.01]) * (HASH_DIMENSION * rank)
    positive = _hashed_indices(rows[0][0])
    negative = _hashed_indices(rows[1][0])
    first = _fm_score(weights, factors, positive, rank)
    if not math.isfinite(first) or first != _fm_score(weights, factors, positive, rank):
        raise ValueError("finite-output or determinism self-test failed")
    difference = first - _fm_score(weights, factors, negative, rank)
    analytic = -1.0 / (1.0 + math.exp(difference))
    step = 1e-6
    shifted = array("d", weights)
    shifted[positive[1]] += step
    shifted_difference = _fm_score(
        shifted, factors, positive, rank
    ) - _fm_score(shifted, factors, negative, rank)
    numeric = (
        math.log1p(math.exp(-shifted_difference))
        - math.log1p(math.exp(-difference))
    ) / step
    if abs(analytic - numeric) > 1e-5:
        raise ValueError("BPR gradient self-test failed")


def _feature_tokens(row: Mapping[str, str]) -> Tuple[str, ...]:
    return tuple("%s=%s" % (name, row[name]) for name in FEATURES)


def _hashed_indices(tokens: Sequence[str]) -> Tuple[int, ...]:
    return tuple(zlib.crc32(token.encode("utf-8")) & 0xFFFF for token in tokens)


def _date_ordinal(value: str) -> int:
    try:
        return datetime.strptime(str(value), "%Y%m%d").date().toordinal()
    except ValueError as error:
        raise ValueError("date must use YYYYMMDD") from error


def _validated_inputs(input_root: Path) -> Tuple[Path, Path]:
    score = _regular_file(input_root / "score.csv")
    parent = _regular_file(input_root / "fm_baseline_predictions.csv")
    digest = _regular_file(
        input_root / "fm_baseline_predictions.sha256"
    ).read_text(encoding="ascii").strip()
    if not _is_sha256(digest) or _sha256_file(parent) != digest:
        raise ValueError("frozen FM prediction identity is invalid")
    return score, parent


def _validate_headers(
    score_fields: Optional[Sequence[str]], parent_fields: Optional[Sequence[str]]
) -> None:
    if not {"row_id", *FEATURES}.issubset(score_fields or ()):
        raise ValueError("score.csv is missing required columns")
    if list(parent_fields or ()) != ["row_id", "user_id", "video_id", "score"]:
        raise ValueError("frozen FM predictions have an invalid header")


def _validate_row_alignment(
    expected: int, score_row: Mapping[str, str], parent_row: Mapping[str, str]
) -> None:
    if int(score_row["row_id"]) != expected or int(parent_row["row_id"]) != expected:
        raise ValueError("candidate rows must be contiguous and ordered")
    if score_row["user_id"] != parent_row["user_id"] or score_row[
        "video_id"
    ] != parent_row["video_id"]:
        raise ValueError("frozen FM predictions do not align with score.csv")
    if not math.isfinite(float(parent_row["score"])):
        raise ValueError("frozen FM predictions must be finite")


def _validate_alignment(score_path: Path, parent_path: Path) -> None:
    with score_path.open(newline="", encoding="utf-8") as score_handle, parent_path.open(
        newline="", encoding="utf-8"
    ) as parent_handle:
        score_rows = csv.DictReader(score_handle, strict=True)
        parent_rows = csv.DictReader(parent_handle, strict=True)
        _validate_headers(score_rows.fieldnames, parent_rows.fieldnames)
        sentinel = object()
        expected = 0
        while True:
            score_row = next(score_rows, sentinel)
            parent_row = next(parent_rows, sentinel)
            if score_row is sentinel and parent_row is sentinel:
                if expected == 0:
                    raise ValueError("score population is empty")
                return
            if score_row is sentinel or parent_row is sentinel:
                raise ValueError("frozen FM predictions have the wrong row count")
            _validate_row_alignment(expected, score_row, parent_row)
            expected += 1


def _empty_diagnostics(formulation: str, seed: Any) -> Dict[str, Any]:
    return {
        "formulation": formulation,
        "seed": int(seed or 0),
        "train_rows": 0,
        "interaction_coverage": 1.0,
        "loss_curve": [0.0],
        "loss_start": 0.0,
        "loss_end": 0.0,
        "pairwise_accuracy": 0.0,
        "gradient_norm": 0.0,
        "residual_mean": 0.0,
        "residual_std": 0.0,
    }


def _write_diagnostics(output_path: Path, diagnostics: Mapping[str, Any]) -> None:
    destination = output_path.with_name(
        "training-diagnostics.json"
        if output_path.name == "predictions.csv"
        else output_path.stem + "-training-diagnostics.json"
    )
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            dict(diagnostics),
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exclusive_copy(source: Path, destination: Path) -> None:
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _regular_file(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("required candidate input is missing")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ValueError("candidate inputs must use canonical paths")
    return resolved
