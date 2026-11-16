import datetime
import random
import unittest

import numpy as np

from datasketch import EventTimeMinHash, EventTimeMinHashFleet, MinHash


def _native_minhash(tokens, **kwargs):
    """Build a native MinHash from a set of tokens (the full rebuild)."""
    m = MinHash(**kwargs)
    for t in tokens:
        m.update(t)
    return m


class TestEventTimeMinHash(unittest.TestCase):
    def test_window_boundaries(self):
        sketch = EventTimeMinHash(window_length=10, num_perm=32)
        sketch.advance_watermark(100)  # window (90, 100]
        sketch.add(b"right-edge", 100)  # inclusive: live
        sketch.add(b"inside", 95)  # live
        sketch.add(b"left-edge", 90)  # exclusive: dropped
        sketch.add(b"older", 10)  # dropped
        sketch.add(b"future", 101)  # pending
        expected = _native_minhash({b"right-edge", b"inside"}, num_perm=32)
        self.assertTrue(np.array_equal(sketch.hashvalues, expected.hashvalues))
        self.assertEqual(len(sketch), 2)
        self.assertEqual(sketch.num_pending, 1)

    def test_late_event_backfilled(self):
        sketch = EventTimeMinHash(window_length=10, num_perm=32)
        sketch.advance_watermark(100)
        before = sketch.hashvalues.copy()
        sketch.add(b"late", 95)  # late but inside the window
        self.assertFalse(np.array_equal(sketch.hashvalues, before))
        expected = _native_minhash({b"late"}, num_perm=32)
        self.assertTrue(np.array_equal(sketch.hashvalues, expected.hashvalues))

    def test_stale_event_dropped(self):
        sketch = EventTimeMinHash(window_length=10, num_perm=32)
        sketch.add(b"x", 100)
        sketch.advance_watermark(100)
        before = sketch.hashvalues.copy()
        sketch.add(b"stale", 90)  # already left of the window
        self.assertTrue(np.array_equal(sketch.hashvalues, before))
        self.assertEqual(len(sketch), 1)

    def test_future_event_takes_effect_at_watermark(self):
        sketch = EventTimeMinHash(window_length=10, num_perm=32)
        sketch.advance_watermark(100)
        sketch.add(b"x", 105)
        self.assertTrue(sketch.is_empty())
        self.assertIsNone(sketch.minhash())
        self.assertEqual(sketch.num_pending, 1)
        sketch.advance_watermark(104)  # still ahead of the watermark
        self.assertTrue(sketch.is_empty())
        sketch.advance_watermark(105)  # window (95, 105] now covers the event
        self.assertFalse(sketch.is_empty())
        self.assertEqual(sketch.num_pending, 0)
        expected = _native_minhash({b"x"}, num_perm=32)
        self.assertTrue(np.array_equal(sketch.hashvalues, expected.hashvalues))

    def test_future_event_skipped_when_window_moved_past(self):
        sketch = EventTimeMinHash(window_length=10, num_perm=32)
        sketch.add(b"x", 100)  # no watermark yet: buffered
        sketch.advance_watermark(50)  # still in the future
        self.assertEqual(sketch.num_pending, 1)
        sketch.advance_watermark(1000)  # window (990, 1000]: event never went live
        self.assertTrue(sketch.is_empty())
        self.assertEqual(sketch.num_pending, 0)
        self.assertIsNone(sketch.minhash())

    def test_same_token_multiple_events_partial_expiry(self):
        sketch = EventTimeMinHash(window_length=10, num_perm=32)
        sketch.advance_watermark(100)
        sketch.add(b"t", 95)
        sketch.add(b"t", 98)
        sketch.add(b"u", 95)
        expected = _native_minhash({b"t", b"u"}, num_perm=32)
        self.assertTrue(np.array_equal(sketch.hashvalues, expected.hashvalues))
        # Expire the events at 95: "t" survives through its event at 98,
        # so its hash values must not be removed; "u" is gone.
        sketch.advance_watermark(106)  # window (96, 106]
        expected = _native_minhash({b"t"}, num_perm=32)
        self.assertTrue(np.array_equal(sketch.hashvalues, expected.hashvalues))
        self.assertEqual(len(sketch), 1)
        # Expire the last occurrence: the sketch returns to empty.
        sketch.advance_watermark(109)  # window (99, 109]
        self.assertTrue(sketch.is_empty())
        self.assertIsNone(sketch.minhash())
        fresh = EventTimeMinHash(window_length=10, num_perm=32)
        self.assertTrue(np.array_equal(sketch.hashvalues, fresh.hashvalues))

    def test_monotonic_watermark(self):
        sketch = EventTimeMinHash(window_length=10)
        sketch.advance_watermark(100)
        sketch.advance_watermark(100)  # equal is allowed
        with self.assertRaises(ValueError):
            sketch.advance_watermark(99)
        with self.assertRaises(ValueError):
            EventTimeMinHash(window_length=0)
        with self.assertRaises(ValueError):
            EventTimeMinHash(window_length=-5)

    def test_exported_minhash_is_native_and_independent(self):
        kwargs = dict(num_perm=64, seed=3)
        sketch = EventTimeMinHash(window_length=10, **kwargs)
        sketch.advance_watermark(100)
        for token in (b"a", b"b", b"c"):
            sketch.add(token, 99)
        exported = sketch.minhash()
        self.assertIsInstance(exported, MinHash)
        expected = _native_minhash({b"a", b"b", b"c"}, **kwargs)
        self.assertEqual(exported, expected)  # native MinHash equality
        self.assertEqual(exported.jaccard(expected), 1.0)
        # Mutating the export must not affect the windowed sketch.
        exported.update(b"zzz")
        self.assertTrue(np.array_equal(sketch.hashvalues, expected.hashvalues))
        self.assertEqual(sketch.minhash(), expected)

    def test_schemes_match_native(self):
        for scheme in ("affine32", "affine64", "legacy"):
            with self.subTest(scheme=scheme):
                kwargs = dict(num_perm=48, seed=11, scheme=scheme)
                sketch = EventTimeMinHash(window_length=10, **kwargs)
                sketch.advance_watermark(100)
                for i in range(20):
                    sketch.add(b"token-%d" % i, 91 + i % 10)
                sketch.advance_watermark(105)  # window (95, 105]
                live = {b"token-%d" % i for i in range(20) if 95 < 91 + i % 10 <= 105}
                exported = sketch.minhash()
                expected = _native_minhash(live, **kwargs)
                self.assertEqual(exported.scheme, scheme)
                self.assertTrue(np.array_equal(exported.hashvalues, expected.hashvalues))

    def test_datetime_event_times(self):
        t0 = datetime.datetime(2026, 11, 16, 12, 0, 0)
        sketch = EventTimeMinHash(window_length=datetime.timedelta(minutes=5), num_perm=32)
        sketch.advance_watermark(t0)
        sketch.add(b"x", t0 - datetime.timedelta(minutes=1))  # late, in window
        sketch.add(b"y", t0 + datetime.timedelta(minutes=2))  # future
        self.assertEqual(len(sketch), 1)
        sketch.advance_watermark(t0 + datetime.timedelta(minutes=2))
        self.assertEqual(len(sketch), 2)
        sketch.advance_watermark(t0 + datetime.timedelta(minutes=10))
        self.assertTrue(sketch.is_empty())


class TestEventTimeMinHashFleet(unittest.TestCase):
    def test_unified_watermark(self):
        kwargs = dict(num_perm=64, seed=5)
        fleet = EventTimeMinHashFleet(window_length=10, **kwargs)
        fleet.add("a", b"shared", 100)
        fleet.add("a", b"only-a", 100)
        fleet.add("b", b"shared", 99)
        fleet.add("b", b"only-b", 101)  # future relative to the first advance
        fleet.advance_watermark(100)  # window (90, 100] for every source
        self.assertEqual(fleet.minhash("a"), _native_minhash({b"shared", b"only-a"}, **kwargs))
        self.assertEqual(fleet.minhash("b"), _native_minhash({b"shared"}, **kwargs))
        # One advance moves every source; b's buffered event goes live.
        fleet.advance_watermark(101)  # window (91, 101]
        self.assertEqual(fleet.minhash("b"), _native_minhash({b"shared", b"only-b"}, **kwargs))

    def test_compare_sources_with_existing_jaccard(self):
        kwargs = dict(num_perm=256, seed=5)
        fleet = EventTimeMinHashFleet(window_length=10, **kwargs)
        fleet.advance_watermark(100)
        for i in range(30):
            fleet.add("a", b"t%d" % i, 95)
            fleet.add("b", b"t%d" % i, 96)
        for i in range(30, 45):
            fleet.add("a", b"t%d" % i, 97)
        ma = fleet.minhash("a")
        mb = fleet.minhash("b")
        # The estimate from the exported sketches equals the estimate of
        # native MinHashes rebuilt from the same window content.
        na = _native_minhash({b"t%d" % i for i in range(45)}, **kwargs)
        nb = _native_minhash({b"t%d" % i for i in range(30)}, **kwargs)
        self.assertEqual(ma.jaccard(mb), na.jaccard(nb))
        self.assertGreater(ma.jaccard(mb), 0.0)
        self.assertLess(ma.jaccard(mb), 1.0)

    def test_unknown_source_returns_none(self):
        fleet = EventTimeMinHashFleet(window_length=10)
        self.assertIsNone(fleet.minhash("nope"))
        self.assertIsNone(fleet.hashvalues("nope"))
        self.assertNotIn("nope", fleet)
        fleet.add("a", b"x", 1)
        self.assertIn("a", fleet)
        self.assertEqual(fleet.sources(), ["a"])
        self.assertEqual(len(fleet), 1)

    def test_late_joining_source_starts_at_current_watermark(self):
        fleet = EventTimeMinHashFleet(window_length=10, num_perm=32)
        fleet.advance_watermark(100)
        fleet.add("late-joiner", b"old", 50)  # older than the window: dropped
        fleet.add("late-joiner", b"new", 100)  # live
        self.assertEqual(len(fleet.sketch("late-joiner")), 1)
        expected = _native_minhash({b"new"}, num_perm=32)
        self.assertTrue(np.array_equal(fleet.hashvalues("late-joiner"), expected.hashvalues))

    def test_fleet_monotonic_watermark(self):
        fleet = EventTimeMinHashFleet(window_length=10)
        fleet.advance_watermark(100)
        with self.assertRaises(ValueError):
            fleet.advance_watermark(50)

    def test_matches_full_rebuild_randomized(self):
        for scheme in ("affine32", "affine64", "legacy"):
            with self.subTest(scheme=scheme):
                self._run_randomized(scheme)

    def _run_randomized(self, scheme):
        rng = random.Random(20261116)
        num_perm, window = 64, 10
        kwargs = dict(num_perm=num_perm, seed=7, scheme=scheme)
        fleet = EventTimeMinHashFleet(window_length=window, **kwargs)
        sources = ["src-a", "src-b", "src-c"]
        tokens = [b"tok-%d" % i for i in range(40)]
        events = {s: [] for s in sources}  # (token, event_time) per source
        now = 0
        fleet.advance_watermark(now)
        for _ in range(600):
            if rng.random() < 0.7:
                src = rng.choice(sources)
                token = rng.choice(tokens)
                # Mix of on-time, late (in and out of window) and future events.
                et = now + rng.randint(-2 * window, 2 * window)
                fleet.add(src, token, et)
                events[src].append((token, et))
            else:
                now += rng.randint(0, 3)
                fleet.advance_watermark(now)
            # After every arrival or advance, every source must equal a full
            # rebuild of its current window content.
            for src in sources:
                live = {t for t, et in events[src] if now - window < et <= now}
                exported = fleet.minhash(src)
                if not live:
                    self.assertIsNone(exported)
                    sketch = fleet.sketch(src)
                    self.assertTrue(sketch is None or sketch.is_empty())
                    continue
                expected = _native_minhash(live, **kwargs)
                self.assertIsNotNone(exported)
                self.assertTrue(np.array_equal(exported.hashvalues, expected.hashvalues))
                self.assertTrue(np.array_equal(fleet.hashvalues(src), expected.hashvalues))
                self.assertEqual(exported.jaccard(expected), 1.0)
                self.assertEqual(len(fleet.sketch(src)), len(live))


if __name__ == "__main__":
    unittest.main()
