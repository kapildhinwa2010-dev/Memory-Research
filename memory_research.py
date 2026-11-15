# Memory-Research - Research framework for associative-memory experiments
# Part of the Kapildhinwa2010-dev Toolkit.

"""MemoryFlow - wave-lattice associative memory with interference recall.

Lineage: reserv.dean-part-2 (helix.py wave arena). Patterns are stored
two ways: as copies for direct correlation recall, and superposed into an
outer-product lattice for Hopfield-style recall that works even after the
copies are dropped.

Honest limitation: lattice recall is only reliable for a handful of
patterns; accuracy decays quickly as the lattice fills.
"""

from __future__ import annotations

import numpy as np

__all__ = ["WaveMemory", "hamming_distance", "noise_pattern"]


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("patterns must have the same shape")
    return int((a != b).sum())


def noise_pattern(pattern: np.ndarray, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Copy of a +/-1 pattern with each bit flipped with probability flip_prob."""
    p = np.asarray(pattern, dtype=np.float64).ravel()
    if not np.all(np.isin(p, (-1.0, 1.0))):
        raise ValueError("pattern must contain only -1 and +1")
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError("flip_prob must be in [0, 1]")
    flips = rng.random(p.size) < flip_prob
    return np.where(flips, -p, p)


def _bipolar(vector: np.ndarray) -> np.ndarray:
    v = np.sign(np.asarray(vector, dtype=np.float64).ravel())
    return np.where(v == 0, 1.0, v)


class WaveMemory:
    """Superposition memory: correlation recall plus true lattice recall."""

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)
        self.lattice = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.patterns: list = []

    def write(self, pattern: np.ndarray) -> None:
        pattern = np.asarray(pattern, dtype=np.float64).ravel()
        if pattern.size != self.dim:
            raise ValueError("pattern size must equal dim")
        if not np.all(np.isin(pattern, (-1.0, 1.0))):
            raise ValueError("patterns must contain only -1 and +1")
        self.lattice += np.outer(pattern, pattern)
        self.patterns.append(pattern.copy())

    def write_many(self, patterns) -> None:
        for pattern in patterns:
            self.write(pattern)

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """Return the stored copy most correlated with the cue."""
        cue = np.asarray(cue, dtype=np.float64).ravel()
        if cue.size != self.dim:
            raise ValueError("cue size must equal dim")
        if not self.patterns:
            raise ValueError("memory is empty")
        scores = [float(cue @ p) for p in self.patterns]
        return self.patterns[int(np.argmax(scores))].copy()

    def recall_from_lattice(self, cue: np.ndarray, steps: int = 5) -> np.ndarray:
        """Hopfield-style recall from the superposed lattice alone.

        Uses only the outer-product lattice (sign updates), so it works
        even when stored copies have been dropped with forget_copies().
        Reliable only for a few patterns; see README.
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")
        state = _bipolar(cue)
        if state.size != self.dim:
            raise ValueError("cue size must equal dim")
        for _ in range(steps):
            state = _bipolar(self.lattice @ state)
        return state

    def forget_copies(self) -> None:
        """Drop stored copies; lattice recall keeps working without them."""
        self.patterns = []

    def cleanup(self, noisy: np.ndarray, max_flips: int = 8) -> np.ndarray:
        """Recall and verify; raise if the result is too far from the cue."""
        recovered = self.recall(noisy)
        flips = hamming_distance(recovered, _bipolar(noisy))
        if flips > max_flips * self.dim // 100:
            raise ValueError("cleanup failed: {} flips".format(flips))
        return recovered

    def __len__(self) -> int:
        return len(self.patterns)
