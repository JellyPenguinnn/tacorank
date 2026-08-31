from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tacorank.context.builder import ContextBuilder
from tacorank.candidate_schema import CANDIDATE_TRAIN_COLUMNS
from tacorank.research.eda import PlannerEdaError, PlannerEdaToolbox
from tacorank.schemas import PlannerDataProfile


def _write_views(root: Path) -> Path:
    root.mkdir()
    (root / "train.csv").write_text(
        "date,time_ms,hourmin,user_id,video_id,author_id,tab,duration_ms,"
        "long_view,is_click,play_time_ms,is_like,is_follow,is_comment,"
        "is_forward,is_hate,is_profile_enter\n"
        "20220408,1,1200,user_secret_a,video_a,author_a,home,100,1,1,90,0,0,0,0,0,0\n"
        "20220408,2,1201,user_secret_a,video_b,author_b,home,200,0,1,50,0,0,0,0,0,0\n"
        "20220409,3,1202,user_b,video_a,author_a,discover,300,1,1,250,0,0,0,0,0,0\n"
        "20220409,4,1203,user_c,video_c,author_c,discover,400,0,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    (root / "score.csv").write_text(
        "row_id,date,user_id,video_id,author_id,tab,duration_ms\n"
        "0,20220410,user_secret_a,video_a,author_a,home,150\n"
        "1,20220410,user_new,video_new,author_c,discover,450\n",
        encoding="utf-8",
    )
    return root


def test_eda_toolbox_builds_deterministic_aggregate_profile(tmp_path: Path) -> None:
    toolbox = PlannerEdaToolbox(_write_views(tmp_path / "candidate-full"))

    first = toolbox.inspect()
    second = toolbox.inspect()

    assert first is second
    assert first.train_rows == 4
    assert first.score_rows == 2
    assert first.train_positive_count == 2
    assert first.train_positive_rate == 0.5
    assert first.train_columns == list(CANDIDATE_TRAIN_COLUMNS)
    assert first.train_date_min == 20220408
    assert first.score_date_max == 20220410
    assert first.train_cardinalities == {
        "user_id": 3,
        "video_id": 3,
        "author_id": 3,
        "tab": 2,
    }
    assert first.train_interactions_per_entity["user_id"].maximum == 2
    assert first.score_entity_overlap["user_id"].seen_in_train_rate == 0.5
    assert first.score_entity_overlap["author_id"].seen_in_train_rate == 1.0
    assert [item.value for item in first.train_long_view_by_date] == [
        "20220408",
        "20220409",
    ]
    assert len(first.profile_sha256) == 64
    serialized = json.dumps(first.model_dump(mode="json"), sort_keys=True)
    assert "user_secret_a" not in serialized
    assert "video_new" not in serialized

    tampered = first.model_dump(mode="json")
    tampered["train_rows"] = 5
    with pytest.raises(ValidationError, match="profile hash"):
        PlannerDataProfile.model_validate(tampered)


def test_eda_toolbox_rejects_a_labelled_score_view(tmp_path: Path) -> None:
    root = _write_views(tmp_path / "candidate-full")
    (root / "score.csv").write_text(
        "row_id,date,user_id,video_id,author_id,tab,duration_ms,long_view\n"
        "0,20220410,user_a,video_a,author_a,home,150,1\n",
        encoding="utf-8",
    )

    with pytest.raises(PlannerEdaError, match="approved schema"):
        PlannerEdaToolbox(root).inspect()


def test_eda_toolbox_accepts_legacy_training_schema_for_resume(
    tmp_path: Path,
) -> None:
    root = _write_views(tmp_path / "candidate-full")
    (root / "train.csv").write_text(
        "date,user_id,video_id,author_id,tab,duration_ms,long_view\n"
        "20220408,user_a,video_a,author_a,home,100,1\n",
        encoding="utf-8",
    )

    profile = PlannerEdaToolbox(root).inspect()

    assert profile.train_rows == 1
    assert profile.train_columns == [
        "date",
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "duration_ms",
        "long_view",
    ]


def test_eda_toolbox_rejects_unapproved_training_columns(tmp_path: Path) -> None:
    root = _write_views(tmp_path / "candidate-full")
    train = root / "train.csv"
    train.write_text(
        train.read_text(encoding="utf-8").replace(
            "is_profile_enter\n", "is_profile_enter,validation_label\n", 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(PlannerEdaError, match="approved schema"):
        PlannerEdaToolbox(root).inspect()


def test_eda_toolbox_rejects_symlinked_input_files(tmp_path: Path) -> None:
    root = _write_views(tmp_path / "candidate-full")
    source = root / "train.csv"
    external = tmp_path / "external-train.csv"
    source.rename(external)
    source.symlink_to(external)

    with pytest.raises(PlannerEdaError, match="regular file"):
        PlannerEdaToolbox(root).inspect()


def test_planner_context_persists_profile_before_provider_call(
    harness, baseline_evaluation, tmp_path: Path
) -> None:
    harness.bootstrap(baseline_evaluation)
    toolbox = PlannerEdaToolbox(_write_views(tmp_path / "candidate-full"))
    harness.context_builder = ContextBuilder(
        harness.config,
        harness.verified_contract,
        harness.context_builder.artifact_store,
        eda_toolbox=toolbox,
    )

    first = harness.context_builder.build_planner(harness.events())
    second = harness.context_builder.build_planner(harness.events())

    assert first.data_profile is not None
    assert first.data_profile.profile_sha256 == second.data_profile.profile_sha256
    assert first.context_id == second.context_id
    assert "Verified aggregate dataset profile" in first.content
    assert "Score rows are unlabeled" in first.content
    assert first.data_profile.train_positive_rate == 0.5
