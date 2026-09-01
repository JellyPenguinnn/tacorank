"""Rank-aware no-op detection.

GAUC and nDCG@5 depend only on within-user ordering, so a candidate that
perturbs every score by a hair while reordering nothing is a no-op in every
way the metrics can see. The score-equality test alone cannot detect that.
"""

from __future__ import annotations

from tacorank.evaluation.no_op import (
    NoOpConfig,
    analyze_prediction_change,
    is_no_op,
    within_user_rank_change,
)


PARENT = [3.0, 1.0, 2.0, 5.0, 4.0, 6.0]
USERS = ["u1", "u1", "u1", "u2", "u2", "u2"]


def test_uniform_tiny_residual_is_a_no_op_despite_changing_every_row():
    # Every score moves, so changed_row_fraction is 1.0, but the ordering
    # inside each user is untouched. This is the shape the run produced.
    candidate = [value + 1e-4 * (index + 1) for index, value in enumerate(PARENT)]
    change = analyze_prediction_change(candidate, PARENT, 1e-12, user_ids=USERS)

    assert change.changed_row_fraction == 1.0
    assert change.within_user_rank_change == 0.0
    assert is_no_op(change, 0.0, NoOpConfig())


def test_real_reordering_is_not_a_no_op():
    candidate = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]  # swaps the first user's top two
    change = analyze_prediction_change(candidate, PARENT, 1e-12, user_ids=USERS)

    assert change.within_user_rank_change > 0.0
    assert not is_no_op(change, 0.0, NoOpConfig())


def test_a_real_gain_is_never_suppressed_as_a_no_op():
    candidate = [value + 1e-9 for value in PARENT]
    change = analyze_prediction_change(candidate, PARENT, 1e-12, user_ids=USERS)

    assert not is_no_op(change, 0.05, NoOpConfig())


def test_cross_user_pairs_are_not_counted():
    # Ordering across users is irrelevant to both metrics: shifting one user's
    # whole list must not register as a change.
    candidate = [value + 100.0 if user == "u2" else value
                 for value, user in zip(PARENT, USERS)]
    assert within_user_rank_change(candidate, PARENT, USERS) == 0.0


def test_single_row_users_contribute_no_pairs():
    assert within_user_rank_change([1.0, 2.0], [2.0, 1.0], ["a", "b"]) is None


def test_grouping_is_required_to_align_with_the_scores():
    try:
        within_user_rank_change([1.0, 2.0], [1.0, 2.0], ["a"])
    except ValueError:
        return
    raise AssertionError("misaligned user ids must be rejected")
