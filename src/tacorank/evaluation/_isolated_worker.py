"""Minimal clean-process runner for the protected official evaluator."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("expected evaluator path and hash")
    evaluator_path = Path(sys.argv[1]).resolve()
    expected_sha256 = sys.argv[2]
    if _sha256_file(evaluator_path) != expected_sha256:
        raise RuntimeError("protected evaluator hash mismatch")

    payload = json.load(sys.stdin)
    module_name = "_tacorank_official_evaluator_%s" % expected_sha256[:12]
    spec = importlib.util.spec_from_file_location(module_name, str(evaluator_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load protected evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        raise RuntimeError("official evaluator has no evaluate function")

    result = evaluate(
        payload["user_ids"],
        payload["labels"],
        payload["scores"],
    )
    json.dump(
        result,
        sys.stdout,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


if __name__ == "__main__":
    main()
