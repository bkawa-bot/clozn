"""_io -- shared atomic JSON write helper for user-data persistence.

Bug this exists to close (round-2 pressure test #1, HIGH: silent total data loss): several stores used
to do `open(path, "w")` then `json.dump(obj, f)` directly against the REAL path. If `obj` turns out to
contain something json can't serialize, `json.dump` raises *after* `open(path, "w")` has already
truncated the file to empty -- so whatever valid data was there is gone, and the next load's
parse-error handler (these stores never raise on load either) silently returns {}/[]  as if the store
had always been empty. One bad `update()`/`set_setting()` call, total data loss, no error surfaced to
the user.

The fix: serialize to a string FIRST via json.dumps (so a non-serializable value raises before the real
file is touched at all), write that string to a temp file in the SAME directory, flush + fsync it, then
atomically replace the real path with os.replace (an atomic rename on both POSIX and Windows -- there is
never a moment where the real path is truncated, empty, or half-written). Any failure before the replace
leaves the prior file completely untouched; any failure is also cleaned up (the temp file removed) rather
than left behind.
"""
from __future__ import annotations

import json
import os
import tempfile


def atomic_write_json(path: str, obj, **dump_kwargs) -> None:
    """Write `obj` as JSON to `path` atomically; raises on failure (callers that want the old
    never-raise contract keep their own try/except around this call -- this function's only job is to
    make sure a failure, of any kind, never destroys the file that was already at `path`).

    `**dump_kwargs` forwards to json.dumps (e.g. indent=2, ensure_ascii=False) so callers keep their
    existing on-disk formatting.
    """
    text = json.dumps(obj, **dump_kwargs)              # raises here FIRST -- path is never touched yet
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-atomic-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)                     # atomic rename: path is either old-and-intact or new-and-complete
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
