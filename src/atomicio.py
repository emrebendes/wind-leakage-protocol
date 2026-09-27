# -*- coding: utf-8 -*-
"""
Atomic file publication, safe under many concurrent writers.
============================================================

Every stage writes results the same way: build the file somewhere temporary,
then rename it into place. The rename is the commit — a reader either sees the
old file or the complete new one, never a half-written one. A crash between
the two costs one redundant unit and corrupts nothing.

The bug this module exists to prevent
-------------------------------------
The obvious temporary name is `path + ".tmp"`, and it is wrong the moment two
workers touch the same output. Both open the identical temporary file, their
writes interleave, and the survivor renames whatever mixture resulted.

On Linux that failure is silent: `os.rename` overwrites an open file happily,
so a corrupted cache is published with no error. On Windows it is loud —
`PermissionError: [WinError 32]` — which is how it was found here, with 832 of
832 precompute units failing at once. The silent Linux behaviour is the more
dangerous of the two, and TRUBA is Linux.

The fix is that the temporary name must be unique per writer, not per target.
PID plus a random token gives that across processes, across nodes on a shared
filesystem, and across retries within one process.

Concurrent renames onto one target
----------------------------------
Two workers may still commit to the same path at nearly the same instant.
POSIX makes that atomic — last writer wins, both files were complete, so
either outcome is correct. Windows can raise while the target is momentarily
open, so the replace is retried briefly before giving up.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from typing import Any, Callable, Iterator


def temp_path(path: str, suffix: str = "") -> str:
    """
    A temporary name unique to this writer.

    Sitting beside the target keeps the rename on one filesystem, which is
    what makes it atomic; a temp directory elsewhere would silently degrade
    into a copy.
    """
    token = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
    return f"{path}.{token}.tmp{suffix}"


def _replace_with_retry(tmp: str, path: str, attempts: int = 20,
                        delay: float = 0.05) -> None:
    """
    Commit the temporary file. Retries the brief window in which another
    process holds the target open — a Windows condition that does not arise
    on POSIX, where the loop simply succeeds first time.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:          # Windows: target in use
            last = exc
            time.sleep(delay * (i + 1))
        except OSError as exc:
            last = exc
            break
    with contextlib.suppress(OSError):
        os.remove(tmp)
    raise OSError(f"could not publish {path}: {last}")


@contextlib.contextmanager
def atomic_path(path: str, suffix: str = "") -> Iterator[str]:
    """
    Yield a temporary path to write to; commit it on a clean exit.

        with atomic_path(out) as tmp:
            np.save(tmp, block)

    `suffix` is for writers that insist on an extension — numpy appends `.npy`
    unless the name already ends in it, and torch.save is indifferent.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = temp_path(path, suffix)
    try:
        yield tmp
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    _replace_with_retry(tmp, path)


def write_json(path: str, obj: Any, indent: int | None = None) -> None:
    """Publish a JSON file atomically."""
    with atomic_path(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent)
    return None


def write_with(path: str, writer: Callable[[str], None],
               suffix: str = "") -> None:
    """
    Publish whatever `writer` puts at the path it is handed.

        write_with(ckpt, lambda p: torch.save(state, p))
    """
    with atomic_path(path, suffix) as tmp:
        writer(tmp)
    return None


def write_npy(path: str, array) -> None:
    """
    Publish a .npy file atomically.

    numpy appends `.npy` to any name lacking it, so the temporary name carries
    that suffix already and the file numpy actually creates is the one renamed.
    """
    import numpy as np

    with atomic_path(path, suffix=".npy") as tmp:
        np.save(tmp, array)
    return None


def write_npz(path: str, **arrays) -> None:
    """Publish a compressed .npz file atomically. See write_npy on suffixes."""
    import numpy as np

    with atomic_path(path, suffix=".npz") as tmp:
        np.savez_compressed(tmp, **arrays)
    return None
