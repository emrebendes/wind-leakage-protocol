# -*- coding: utf-8 -*-
"""
Job ledger tests.
=================

The ledger is what makes a three-day job resumable, so its failure modes are
quiet by nature: nothing crashes, the numbers just stop meaning what they say.
These tests pin the behaviour that a run depends on.

The case that motivated the file: a stage invoked with `--methods none dwt`
claimed the VMD units too, could not execute them, and marked them DONE. The
stage then reported 7488/7488 finished having computed 832. Nothing errored,
and no later stage could tell that most of the cache was missing.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from jobstate import JobLedger                                    # noqa: E402


@pytest.fixture
def ledger(tmp_path):
    return JobLedger(str(tmp_path / "state.db"), "demo")


def _keys(prefix, n):
    return [f"{prefix}|{i}" for i in range(n)]


# =============================================================================
def test_claims_run_out(ledger):
    ledger.register_many(_keys("a", 3))
    seen = []
    while (u := ledger.claim()) is not None:
        seen.append(u.key)
        ledger.done(u.key)
    assert sorted(seen) == sorted(_keys("a", 3))
    assert ledger.counts()["DONE"] == 3


def test_scope_hides_other_invocations_units(ledger):
    """A scoped worker must never be handed work it cannot do."""
    ledger.register_many(_keys("dwt", 4) + _keys("vmd", 4))
    ledger.set_scope(_keys("dwt", 4))

    claimed = []
    while (u := ledger.claim()) is not None:
        claimed.append(u.key)
        ledger.done(u.key)

    assert sorted(claimed) == sorted(_keys("dwt", 4))
    assert all(k.startswith("dwt") for k in claimed), \
        "a scoped worker was handed a unit outside its scope"


def test_scope_leaves_other_units_pending(tmp_path):
    """
    The units this invocation skipped must still be waiting for the one that
    covers them. This is the regression: they used to come back DONE.
    """
    db = str(tmp_path / "state.db")
    first = JobLedger(db, "demo")
    first.register_many(_keys("dwt", 3) + _keys("vmd", 3))
    first.set_scope(_keys("dwt", 3))
    while (u := first.claim()) is not None:
        first.done(u.key)

    audit = JobLedger(db, "demo")
    c = audit.counts()
    assert c["DONE"] == 3, f"expected 3 done, got {c['DONE']}"
    assert c["PENDING"] == 3, "the VMD units were not left for a later run"

    second = JobLedger(db, "demo")
    second.set_scope(_keys("vmd", 3))
    did = []
    while (u := second.claim()) is not None:
        did.append(u.key)
        second.done(u.key)
    assert sorted(did) == sorted(_keys("vmd", 3))
    assert audit.counts()["DONE"] == 6


def test_counts_follow_the_scope(ledger):
    """Progress must be reported against the work asked for, not the stage."""
    ledger.register_many(_keys("dwt", 2) + _keys("vmd", 8))
    ledger.set_scope(_keys("dwt", 2))
    assert ledger.counts()["TOTAL"] == 2, \
        "a subset run reported the whole stage's unit count"


def test_release_returns_a_unit_without_consuming_an_attempt(ledger):
    ledger.register_many(["solo"])
    u = ledger.claim()
    assert u is not None and u.attempts == 1

    ledger.release("solo")
    assert ledger.counts()["PENDING"] == 1

    again = ledger.claim()
    assert again is not None
    assert again.attempts == 1, "release consumed one of the unit's attempts"


def test_failed_units_are_retried_then_given_up_on(tmp_path):
    led = JobLedger(str(tmp_path / "s.db"), "demo", max_attempts=3)
    led.register_many(["flaky"])
    for _ in range(3):
        u = led.claim()
        assert u is not None
        led.fail(u.key, "boom")
    assert led.claim() is None, "a unit was retried past max_attempts"
    assert led.counts()["FAILED"] == 1


def test_reset_makes_work_claimable_again(ledger):
    ledger.register_many(["x"])
    u = ledger.claim()
    ledger.fail(u.key, "boom")
    ledger.force_reset()
    assert ledger.claim() is not None


# =============================================================================
def test_done_units_with_missing_output_are_reopened(tmp_path):
    """
    A unit is only finished if its output still exists.

    Deleting a results directory without resetting the ledger leaves every unit
    DONE and its output gone. The stage then reports complete and the missing
    runs are simply absent from the analysis — which is what happened:
    `results_causal` was removed, the deploy ledger was not, and one of two
    units never re-ran while the stage printed "done 2/2".
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
    from pipeline import Stage

    outputs = tmp_path / "out"
    outputs.mkdir()
    (outputs / "b.txt").write_text("kept")

    class Demo(Stage):
        name = "demo"
        ran = []

        def plan(self):
            return {"a": {"f": "a.txt"}, "b": {"f": "b.txt"}}

        def output_exists(self, key, spec):
            return (outputs / spec["f"]).exists()

        def execute(self, key, spec):
            (outputs / spec["f"]).write_text("made")
            Demo.ran.append(key)
            return None

    db = str(tmp_path / "s.db")
    stage = Demo(db, verbose=False)

    # Both units recorded as finished, but only b's output is on disk.
    stage.ledger.register_many(["a", "b"])
    stage.ledger.done("a")
    stage.ledger.done("b")

    stage.run()

    assert Demo.ran == ["a"], \
        f"expected only the unit with a missing output to re-run, got {Demo.ran}"
    assert (outputs / "a.txt").exists()


def test_units_with_intact_output_are_not_repeated(tmp_path):
    """The check must not undo resumability by re-running finished work."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
    from pipeline import Stage

    outputs = tmp_path / "out"
    outputs.mkdir()
    (outputs / "a.txt").write_text("kept")

    class Demo(Stage):
        name = "demo2"
        ran = []

        def plan(self):
            return {"a": {"f": "a.txt"}}

        def output_exists(self, key, spec):
            return (outputs / spec["f"]).exists()

        def execute(self, key, spec):
            Demo.ran.append(key)
            return None

    stage = Demo(str(tmp_path / "s.db"), verbose=False)
    stage.ledger.register_many(["a"])
    stage.ledger.done("a")
    stage.run()

    assert Demo.ran == [], "a finished unit with its output intact was re-run"
