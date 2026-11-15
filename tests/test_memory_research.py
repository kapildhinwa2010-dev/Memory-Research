"""Tests for Memory-Research: true lattice recall and noise."""

import numpy as np
import pytest

from memory_research import WaveMemory, hamming_distance, noise_pattern


def test_write_and_correlation_recall():
    memory = WaveMemory(dim=8)
    p = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    memory.write(p)
    assert np.array_equal(memory.recall(p), p)
    assert len(memory) == 1


def test_lattice_recall_works_without_copies():
    memory = WaveMemory(dim=16)
    p1 = np.array([1, -1] * 8, dtype=float)
    p2 = np.array([-1, 1] * 4 + [1, -1] * 4, dtype=float)
    memory.write(p1)
    memory.write(p2)
    memory.forget_copies()
    assert len(memory) == 0
    assert np.array_equal(memory.recall_from_lattice(p1), p1)
    assert np.array_equal(memory.recall_from_lattice(p2), p2)


def test_noise_pattern_flips_bits():
    rng = np.random.default_rng(0)
    p = np.ones(1000)
    noisy = noise_pattern(p, flip_prob=0.1, rng=rng)
    flips = hamming_distance(noisy, p)
    assert 50 < flips < 150
    assert np.all(np.isin(noisy, (-1.0, 1.0)))
    assert np.array_equal(noise_pattern(p, 0.0, rng), p)


def test_noise_pattern_validates():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        noise_pattern(np.ones(4), 1.5, rng)
    with pytest.raises(ValueError):
        noise_pattern(np.zeros(4), 0.1, rng)


def test_cleanup_recovers_small_noise():
    memory = WaveMemory(dim=32)
    p = np.array([1, -1] * 16, dtype=float)
    memory.write(p)
    rng = np.random.default_rng(1)
    noisy = noise_pattern(p, flip_prob=0.05, rng=rng)
    recovered = memory.cleanup(noisy, max_flips=10)
    assert np.array_equal(recovered, p)


def test_hamming_distance_validates():
    with pytest.raises(ValueError):
        hamming_distance(np.ones(4), np.ones(5))
