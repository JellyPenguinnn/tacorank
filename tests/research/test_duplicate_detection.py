from types import SimpleNamespace

from tacorank.research.duplicate_detection import DuplicateDetector, compute_duplicate_key


def make_spec(**overrides):
    values = dict(
        parent_commit_sha="abc",
        parent_experiment_id="exp_0000",
        family="objective",
        target_stage="loss",
        target_files=["solution/loss.py"],
        fidelity_plan=["smoke", "proxy", "full"],
        change_summary="Use pairwise BPR loss",
        method_card_ids=["objective_pairwise_bpr"],
    )
    values.update(overrides)
    spec = SimpleNamespace(**values)
    if not hasattr(spec, "duplicate_key"):
        spec.duplicate_key = compute_duplicate_key(spec)
    return spec


def test_duplicate_key_is_stable_under_whitespace_and_file_order():
    first = make_spec(target_files=["solution/loss.py", "solution/model.py"])
    second = make_spec(
        target_files=["solution/model.py", "solution/loss.py"],
        change_summary="  USE pairwise   BPR loss ",
    )

    assert compute_duplicate_key(first) == compute_duplicate_key(second)


def test_detector_rejects_seen_key_and_validates_supplied_key():
    first = make_spec()
    detector = DuplicateDetector([first])
    second = make_spec()

    assert detector.contains(second)
    assert detector.validate(second)
    assert not detector.validate(make_spec(duplicate_key="wrong"))
