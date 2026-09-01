"""Derived views must survive a concurrent reader on Windows.

os.replace over a file another handle has open is denied on Windows and
succeeds on POSIX. run_20260831T043328Z died mid-experiment on exactly that:

  PermissionError: [WinError 5] Access is denied:
  'runs/<id>/state.json.tmp' -> 'runs/<id>/state.json'

Anything reading a derived view concurrently can cause it -- a status watcher,
an editor, a virus scanner, the search indexer -- and the holder releases in
milliseconds, so the write must retry rather than abort the run.
"""

from __future__ import annotations

import threading
import time

from tacorank.reporting.results import _atomic_write


def test_replace_succeeds_while_a_reader_holds_the_file(tmp_path):
    target = tmp_path / "state.json"
    target.write_text('{"generation": 1}', encoding="utf-8")

    started = threading.Event()

    def hold():
        with target.open("r", encoding="utf-8") as handle:
            handle.read()
            started.set()
            time.sleep(0.25)

    reader = threading.Thread(target=hold)
    reader.start()
    started.wait(timeout=5)
    try:
        _atomic_write(target, '{"generation": 2}')
    finally:
        reader.join()

    assert target.read_text(encoding="utf-8") == '{"generation": 2}'
    assert not (tmp_path / "state.json.tmp").exists()


def test_uncontended_write_still_replaces(tmp_path):
    target = tmp_path / "status.md"
    _atomic_write(target, "first")
    _atomic_write(target, "second")

    assert target.read_text(encoding="utf-8") == "second"
