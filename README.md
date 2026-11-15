# Memory-Research

[![CI](https://github.com/kapildhinwa2010-dev/Memory-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Memory-Research/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Memory-Research/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Memory-Research/actions/workflows/codeql.yml)

Research framework for associative memory: correlation recall plus true lattice recall.

## What it does

WaveMemory stores +/-1 patterns two ways: as copies for direct
correlation recall, and superposed into an outer-product lattice for
Hopfield-style sign-update recall. recall_from_lattice() uses only the
lattice, so it keeps working after forget_copies() drops the copies.

    from memory_research import WaveMemory, noise_pattern

    memory = WaveMemory(dim=64)
    memory.write(pattern)
    memory.forget_copies()
    recovered = memory.recall_from_lattice(noisy_cue)

## API

- WaveMemory.write / write_many / recall / recall_from_lattice /
  forget_copies / cleanup
- noise_pattern(pattern, flip_prob, rng) - bit flips at a given rate
- hamming_distance(a, b)

## Honest results and limitations

- Lattice recall is only reliable for a handful of patterns; accuracy
  decays quickly as the lattice fills (classic Hopfield behaviour,
  roughly 0.138*dim patterns at best, sooner in practice).
- cleanup() judges success by distance from the cue sign, not by
  whether the true pattern was recovered.
- Negative result from the source experiments: under heavier noise the
  carrier-mode recall variant failed about half the time; the lattice
  design here is the safer one but still not robust at high noise.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
