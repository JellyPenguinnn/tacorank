"""Contract-driven candidate data-access boundary checks."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from .path_policy import PolicyViolation, ViolationCode


@dataclass(frozen=True)
class DataViewPolicy:
    """Allowed columns and locations for one candidate-visible data view."""

    view_id: str
    allowed_columns: Tuple[str, ...]
    allowed_path_prefixes: Tuple[str, ...]


class DataAccessPolicy:
    """Enforce frozen column/path rules without knowing metric or target names."""

    def __init__(
        self,
        *,
        views: Sequence[DataViewPolicy],
        protected_columns: Sequence[str],
        hidden_path_tokens: Sequence[str],
        future_column_patterns: Sequence[str],
        approved_future_columns: Sequence[str] = (),
    ) -> None:
        self._views = {view.view_id: view for view in views}
        if len(self._views) != len(views):
            raise ValueError("data view ids must be unique")
        self.protected_columns = frozenset(protected_columns)
        self.hidden_path_tokens = tuple(token.casefold() for token in hidden_path_tokens)
        try:
            self.future_column_patterns = tuple(
                re.compile(pattern, re.IGNORECASE) for pattern in future_column_patterns
            )
        except re.error as exc:
            raise ValueError("invalid future-column pattern: {}".format(exc))
        self.approved_future_columns = frozenset(approved_future_columns)

    def validate_access(
        self,
        view_id: str,
        *,
        paths: Sequence[str],
        columns: Sequence[str],
    ) -> Tuple[PolicyViolation, ...]:
        view = self._views.get(view_id)
        if view is None:
            return (
                PolicyViolation(
                    ViolationCode.HIDDEN_LABEL_ACCESS,
                    "data_boundary",
                    "candidate requested an unknown or protected data view",
                ),
            )
        findings = []
        allowed_columns = set(view.allowed_columns)
        for column in columns:
            if column in self.protected_columns or column not in allowed_columns:
                findings.append(
                    PolicyViolation(
                        ViolationCode.HIDDEN_LABEL_ACCESS,
                        "data_boundary",
                        "candidate requested a column outside the approved data view",
                    )
                )
            if self._is_future_column(column):
                findings.append(
                    PolicyViolation(
                        ViolationCode.FUTURE_INFORMATION_LEAKAGE,
                        "data_boundary",
                        "candidate requested a future-information column",
                    )
                )
        for path in paths:
            folded = path.casefold()
            if any(token in folded for token in self.hidden_path_tokens):
                findings.append(
                    PolicyViolation(
                        ViolationCode.HIDDEN_LABEL_ACCESS,
                        "data_boundary",
                        "candidate requested a protected label/data location",
                        path,
                    )
                )
            elif not any(
                path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")
                for prefix in view.allowed_path_prefixes
            ):
                findings.append(
                    PolicyViolation(
                        ViolationCode.HIDDEN_LABEL_ACCESS,
                        "data_boundary",
                        "candidate requested a path outside the approved data view",
                        path,
                    )
                )
        return _deduplicate(findings)

    def inspect_source(self, source: str, path: str) -> Tuple[PolicyViolation, ...]:
        """Inspect names/string literals in valid Python without reading data."""

        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return ()
        findings = []
        for node in ast.walk(tree):
            value = _candidate_literal(node)
            if value is None:
                continue
            folded = value.casefold()
            protected_column_literal = (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and value in self.protected_columns
            )
            if protected_column_literal or any(
                token in folded for token in self.hidden_path_tokens
            ):
                findings.append(
                    PolicyViolation(
                        ViolationCode.HIDDEN_LABEL_ACCESS,
                        "data_boundary",
                        "candidate source references a protected label or data location at line {}".format(
                            getattr(node, "lineno", "?")
                        ),
                        path,
                    )
                )
            if self._is_future_column(value):
                findings.append(
                    PolicyViolation(
                        ViolationCode.FUTURE_INFORMATION_LEAKAGE,
                        "data_boundary",
                        "candidate source references future information at line {}".format(
                            getattr(node, "lineno", "?")
                        ),
                        path,
                    )
                )
        return _deduplicate(findings)

    def _is_future_column(self, value: str) -> bool:
        return value not in self.approved_future_columns and any(
            pattern.search(value) for pattern in self.future_column_patterns
        )


def _candidate_literal(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _deduplicate(findings: Iterable[PolicyViolation]) -> Tuple[PolicyViolation, ...]:
    unique = {}
    for finding in findings:
        key = (finding.code.value, finding.check, finding.path, finding.message)
        unique[key] = finding
    order = sorted(unique, key=lambda item: tuple(part or "" for part in item))
    return tuple(unique[key] for key in order)
