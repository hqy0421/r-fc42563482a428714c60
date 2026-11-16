from __future__ import annotations

import heapq
import itertools
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np

from datasketch.minhash import _SCHEME_MAX_HASHES, MinHash


def _check_window_length(window_length) -> None:
    """Validate that window_length is a positive duration.

    ``window_length - window_length`` is the zero of whatever type the caller
    uses for event times (a number for UNIX timestamps, ``datetime.timedelta``
    for ``datetime``), so the check stays agnostic of the time type.
    """
    try:
        positive = window_length > window_length - window_length
    except TypeError as err:
        raise ValueError(
            "window_length must be a positive duration comparable with the event times "
            "(e.g. a number of seconds for UNIX timestamps, datetime.timedelta for datetimes)"
        ) from err
    if not positive:
        raise ValueError("window_length must be positive")


class EventTimeMinHash:
    """EventTimeMinHash maintains the MinHash of one source's content inside
    an event-time sliding window ``(watermark - window_length, watermark]``.

    The window slides over *event time*, not arrival time:

    - The caller advances the watermark monotonically with
      :meth:`advance_watermark`; only events with
      ``watermark - window_length < event_time <= watermark`` are live.
    - Late events that still fall inside the window are backfilled into the
      sketch on arrival; events already older than the window are dropped.
    - Events with ``event_time > watermark`` are buffered and take effect only
      once the watermark passes their event time (and are dropped unseen if
      the window has already moved past them by then).
    - The same token may occur in many events: occurrences are reference
      counted, so expiring one event never removes a token still contributed
      by other live events.

    After every :meth:`add` and :meth:`advance_watermark`, :attr:`hashvalues`
    equals the :attr:`datasketch.MinHash.hashvalues` of a native
    :class:`datasketch.MinHash` (same ``num_perm``, ``seed``, ``hashfunc`` and
    ``scheme``) built from the set of distinct live tokens. This is maintained
    incrementally: per permutation slot the sketch keeps the multiplicity of
    every hash value currently in the window, so :attr:`hashvalues` is always
    the per-slot minimum and deletions never replay the window. Memory is
    proportional to ``num_perm`` times the number of distinct live tokens.

    Use :meth:`minhash` to export an independent native
    :class:`datasketch.MinHash` of the current window (``None`` when the
    window is empty) and compare sources with the existing
    :meth:`datasketch.MinHash.jaccard`.

    Args:
        window_length: Length of the sliding window, in the same units as the
            event times (e.g. seconds for UNIX timestamps, or
            ``datetime.timedelta`` for ``datetime`` event times). Must be
            positive.
        num_perm (int): Number of random permutation functions, same meaning
            as in :class:`datasketch.MinHash`.
        seed (int): Random seed for the permutation functions, same meaning
            as in :class:`datasketch.MinHash`. Sketches compared with each
            other must use the same seed.
        hashfunc (Optional[Callable]): Hash function, same meaning as in
            :class:`datasketch.MinHash`. Defaults to the SHA1-based function
            of the chosen scheme, which requires ``bytes`` tokens.
        scheme (Optional[str]): Permutation scheme forwarded to
            :class:`datasketch.MinHash` (``"affine32"`` by default).
        watermark: Initial watermark. If None (default), nothing is live
            until the first :meth:`advance_watermark` call.

    Attributes:
        hashvalues (numpy.ndarray): Hash values of the current window, kept
            consistent with a full rebuild of the window content.
        watermark: The current watermark (None until the first advance).
        window_length: The sliding window length.

    Note:
        Tokens must be hashable (they key the reference counts) and accepted
        by ``hashfunc`` -- ``bytes`` with the default hash functions, exactly
        like :meth:`datasketch.MinHash.update`.

    Example:
        .. code-block:: python

            from datasketch import EventTimeMinHash

            sketch = EventTimeMinHash(window_length=3600, num_perm=128)
            sketch.advance_watermark(1_700_003_600)  # window (1_700_000_000, 1_700_003_600]
            sketch.add(b"token-1", 1_700_003_600)  # live: right edge is inclusive
            sketch.add(b"token-2", 1_700_000_100)  # live: late but inside the window
            sketch.add(b"token-3", 1_700_000_000)  # dropped: already left of the window
            sketch.add(b"token-4", 1_700_007_200)  # buffered: ahead of the watermark
            sketch.advance_watermark(1_700_007_200)  # token-4 goes live, token-2 expires

            current = sketch.minhash()
            if current is not None:
                current.jaccard(other_minhash)

    """

    def __init__(
        self,
        window_length,
        num_perm: int = 128,
        seed: int = 1,
        hashfunc: Optional[Callable] = None,
        scheme: Optional[str] = None,
        watermark=None,
    ) -> None:
        _check_window_length(window_length)
        self.window_length = window_length
        # A never-updated MinHash used as the template: it owns the
        # permutation parameters, the hash function and the scheme, so this
        # sketch and everything exported from it stay bit-compatible with a
        # natively built MinHash over the same content.
        self._template = MinHash(num_perm=num_perm, seed=seed, hashfunc=hashfunc, scheme=scheme)
        self.hashvalues = self._template.hashvalues.copy()
        self.watermark = watermark
        # token -> number of live events currently contributing the token.
        self._counts: Dict[Hashable, int] = {}
        # Per permutation slot: hash value -> number of distinct live tokens
        # hashing to it in that slot. hashvalues[i] is always the smallest key
        # of slot i (or the scheme's empty sentinel when the slot is empty);
        # this is what makes deletions exact without replaying the window.
        self._slot_counts: List[Dict[int, int]] = [dict() for _ in range(self._template.num_perm)]
        # Live and future events as (event_time, sequence, token) min-heaps;
        # the sequence number keeps the heap from comparing tokens.
        self._live: List[Tuple] = []
        self._pending: List[Tuple] = []
        self._seq = itertools.count()

    @property
    def num_perm(self) -> int:
        """int: Number of permutation functions."""
        return self._template.num_perm

    @property
    def seed(self) -> int:
        """int: The random seed of the permutation functions."""
        return self._template.seed

    @property
    def scheme(self) -> str:
        """str: The permutation scheme."""
        return self._template.scheme

    @property
    def hashfunc(self) -> Callable:
        """Callable: The hash function used for tokens."""
        return self._template.hashfunc

    def add(self, token, event_time) -> None:
        """Add one occurrence of ``token`` observed at ``event_time``.

        The occurrence becomes live immediately if it falls inside the
        current window (late data is backfilled), is buffered if it lies
        ahead of the watermark, and is dropped if it is already older than
        the window.

        Args:
            token: The token; hashable and accepted by ``hashfunc`` (``bytes``
                with the default hash functions).
            event_time: Event time of the occurrence, comparable with the
                watermark.

        """
        watermark = self.watermark
        if watermark is None or event_time > watermark:
            heapq.heappush(self._pending, (event_time, next(self._seq), token))
        elif event_time > watermark - self.window_length:
            self._make_live(token, event_time)
        # Otherwise the event is already left of the window: drop it.

    def advance_watermark(self, watermark) -> None:
        """Advance the watermark, activating and expiring events.

        Buffered events with ``event_time <= watermark`` enter the window
        (unless it has already moved past them), and live events with
        ``event_time <= watermark - window_length`` leave it.

        Args:
            watermark: The new watermark; must not be smaller than the
                current one.

        Raises:
            ValueError: If ``watermark`` is smaller than the current
                watermark (watermarks must advance monotonically).

        """
        if self.watermark is not None and watermark < self.watermark:
            raise ValueError(
                "watermark must advance monotonically: %r is before the current watermark %r"
                % (watermark, self.watermark)
            )
        self.watermark = watermark
        cutoff = watermark - self.window_length
        # Activate buffered events the watermark has reached; events already
        # left of the window never become live.
        pending = self._pending
        while pending and pending[0][0] <= watermark:
            event_time, _, token = heapq.heappop(pending)
            if event_time > cutoff:
                self._make_live(token, event_time)
        # Expire live events that fell out of the left edge of the window.
        live = self._live
        counts = self._counts
        while live and live[0][0] <= cutoff:
            _, _, token = heapq.heappop(live)
            remaining = counts[token] - 1
            if remaining:
                counts[token] = remaining
            else:
                del counts[token]
                self._remove_token(token)

    def minhash(self) -> Optional[MinHash]:
        """Export the current window as an independent native MinHash.

        Returns:
            Optional[datasketch.MinHash]: A new :class:`datasketch.MinHash`
                (same ``num_perm``, ``seed``, ``hashfunc`` and ``scheme``)
                whose hash values equal a full rebuild of the current window
                content, or None if the window has no live content. The
                returned sketch is independent: mutating it does not affect
                this sketch.

        """
        if not self._counts:
            return None
        exported = self._template.copy()
        exported.hashvalues = self.hashvalues.copy()
        return exported

    def is_empty(self) -> bool:
        """Returns:
        bool: True if the current window has no live content.
        """
        return not self._counts

    @property
    def num_pending(self) -> int:
        """int: Number of buffered events waiting for the watermark."""
        return len(self._pending)

    def __len__(self) -> int:
        """Returns:
        int: Number of distinct live tokens in the current window.
        """
        return len(self._counts)

    def _make_live(self, token, event_time) -> None:
        count = self._counts.get(token, 0)
        self._counts[token] = count + 1
        heapq.heappush(self._live, (event_time, next(self._seq), token))
        if count:
            # Already contributed by another live event: hash values unchanged.
            return
        phv = self._template._permuted_hashes(token)
        self.hashvalues = np.minimum(self.hashvalues, phv)
        for i in range(self._template.num_perm):
            slot = self._slot_counts[i]
            value = int(phv[i])
            slot[value] = slot.get(value, 0) + 1

    def _remove_token(self, token) -> None:
        phv = self._template._permuted_hashes(token)
        max_hash = _SCHEME_MAX_HASHES[self._template.scheme]
        hashvalues = self.hashvalues
        for i in range(self._template.num_perm):
            slot = self._slot_counts[i]
            value = int(phv[i])
            remaining = slot[value] - 1
            if remaining:
                slot[value] = remaining
                continue
            del slot[value]
            if hashvalues[i] == value:
                # The minimum of this slot left the window; rescan this slot
                # only, never the whole window.
                hashvalues[i] = min(slot) if slot else max_hash


class EventTimeMinHashFleet:
    """EventTimeMinHashFleet maintains one :class:`EventTimeMinHash` per
    source under a single watermark, so the recent content of many sources
    can be compared over the same event-time window
    ``(watermark - window_length, watermark]``.

    The caller feeds events with :meth:`add` and advances the unified
    watermark monotonically with :meth:`advance_watermark`; after either,
    every source's hash values equal a full rebuild of that source's current
    window. Export any source's window as an independent native
    :class:`datasketch.MinHash` with :meth:`minhash` (``None`` when the
    source is unknown or has no live content) and compare sources with the
    existing :meth:`datasketch.MinHash.jaccard`.

    Args:
        window_length: Length of the sliding window shared by all sources,
            in the same units as the event times. Must be positive.
        num_perm (int): Number of random permutation functions of every
            per-source sketch.
        seed (int): Random seed shared by every per-source sketch, so their
            exported MinHashes are comparable.
        hashfunc (Optional[Callable]): Hash function of every per-source
            sketch (default: the SHA1-based function of the scheme).
        scheme (Optional[str]): Permutation scheme of every per-source
            sketch (``"affine32"`` by default).

    Attributes:
        watermark: The current unified watermark (None until the first
            advance).
        window_length: The sliding window length shared by all sources.

    Example:
        .. code-block:: python

            from datasketch import EventTimeMinHashFleet

            fleet = EventTimeMinHashFleet(window_length=3600, num_perm=128)
            fleet.add("crawler-a", b"token-1", 1_700_003_000)
            fleet.add("crawler-b", b"token-1", 1_700_003_100)
            fleet.advance_watermark(1_700_003_600)

            ma = fleet.minhash("crawler-a")
            mb = fleet.minhash("crawler-b")
            if ma is not None and mb is not None:
                similarity = ma.jaccard(mb)

    """

    def __init__(
        self,
        window_length,
        num_perm: int = 128,
        seed: int = 1,
        hashfunc: Optional[Callable] = None,
        scheme: Optional[str] = None,
    ) -> None:
        _check_window_length(window_length)
        self.window_length = window_length
        self._sketch_kwargs = {"num_perm": num_perm, "seed": seed, "hashfunc": hashfunc, "scheme": scheme}
        self.watermark = None
        self._sources: Dict[Hashable, EventTimeMinHash] = {}

    def add(self, source, token, event_time) -> None:
        """Add one occurrence of ``token`` at ``event_time`` to ``source``.

        Sources are created on first use; a source created after the
        watermark has advanced starts at the current watermark.

        Args:
            source: Hashable identifier of the source.
            token: The token; hashable and accepted by the hash function
                (``bytes`` with the default hash functions).
            event_time: Event time of the occurrence, comparable with the
                watermark.

        """
        sketch = self._sources.get(source)
        if sketch is None:
            sketch = EventTimeMinHash(window_length=self.window_length, watermark=self.watermark, **self._sketch_kwargs)
            self._sources[source] = sketch
        sketch.add(token, event_time)

    def advance_watermark(self, watermark) -> None:
        """Advance the unified watermark of all sources.

        Args:
            watermark: The new watermark; must not be smaller than the
                current one.

        Raises:
            ValueError: If ``watermark`` is smaller than the current
                watermark (watermarks must advance monotonically).

        """
        if self.watermark is not None and watermark < self.watermark:
            raise ValueError(
                "watermark must advance monotonically: %r is before the current watermark %r"
                % (watermark, self.watermark)
            )
        self.watermark = watermark
        for sketch in self._sources.values():
            sketch.advance_watermark(watermark)

    def minhash(self, source) -> Optional[MinHash]:
        """Export ``source``'s current window as an independent native
        MinHash.

        Returns:
            Optional[datasketch.MinHash]: The exported MinHash, or None if
                the source is unknown or its window has no live content.

        """
        sketch = self._sources.get(source)
        return sketch.minhash() if sketch is not None else None

    def hashvalues(self, source) -> Optional[np.ndarray]:
        """Return a copy of ``source``'s current hash values.

        Returns:
            Optional[numpy.ndarray]: The hash values, or None if the source
                is unknown.

        """
        sketch = self._sources.get(source)
        return sketch.hashvalues.copy() if sketch is not None else None

    def sketch(self, source) -> Optional[EventTimeMinHash]:
        """Return ``source``'s underlying :class:`EventTimeMinHash`, or None
        if the source is unknown.
        """
        return self._sources.get(source)

    def sources(self) -> List:
        """Return the list of known sources."""
        return list(self._sources)

    def __contains__(self, source) -> bool:
        return source in self._sources

    def __len__(self) -> int:
        """Returns:
        int: Number of known sources.
        """
        return len(self._sources)
