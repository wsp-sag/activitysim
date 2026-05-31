from __future__ import annotations

import argparse
import gc
import logging
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activitysim.core import workflow


STEP_NAME = "compare_eet_cache"


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("activitysim.core.random").setLevel(logging.DEBUG)


def make_rng(workdir: Path, cache_enabled: bool, rows: int):
    workdir.joinpath("configs").mkdir(parents=True, exist_ok=True)
    workdir.joinpath("data").mkdir(parents=True, exist_ok=True)
    workdir.joinpath("configs", "settings.yaml").write_text("# empty\n")

    state = workflow.State.make_default(workdir, cache_dir=workdir.joinpath("cache"))
    state.settings.rng_cache_enabled = cache_enabled
    state._initialize_prng(base_seed=0)

    choosers = pd.DataFrame(
        {"dummy": 1},
        index=pd.Index(range(rows), name="person_id"),
    )

    state.rng().begin_step(STEP_NAME)
    state.rng().add_channel("persons", choosers)
    return state, choosers


def build_utility_frame(rows: int, alternative_count: int):
    return pd.DataFrame(
        0.0,
        index=pd.Index(range(rows), name="person_id"),
        columns=pd.Index(range(alternative_count), name="alternative_id"),
    )


def run_sampled_eet(state, utilities, sample_size: int, alternative_count: int):
    last_rands = None
    for _ in range(sample_size):
        last_rands = state.rng().gumbel_for_df(utilities, n=alternative_count)
    return None if last_rands is None else np.array(last_rands, copy=True)


def timed_uncached_sampled(rows: int, alternative_count: int, sample_size: int, repeats: int):
    timings = []
    with tempfile.TemporaryDirectory() as tmpdir:
        state, _choosers = make_rng(Path(tmpdir) / "uncached", cache_enabled=False, rows=rows)
        utilities = build_utility_frame(rows, alternative_count)
        for _ in range(repeats):
            state.rng().end_step(STEP_NAME)
            state.rng().begin_step(STEP_NAME)
            t0 = time.perf_counter()
            sample = run_sampled_eet(state, utilities, sample_size, alternative_count)
            timings.append((time.perf_counter() - t0, sample))
        state.rng().end_step(STEP_NAME)
        del utilities
        del state
        gc.collect()
    return timings


def timed_cached_loads(rows: int, alternative_count: int, sample_size: int, repeats: int):
    timings = []
    with tempfile.TemporaryDirectory() as tmpdir:
        state, _choosers = make_rng(Path(tmpdir) / "cached", cache_enabled=True, rows=rows)
        utilities = build_utility_frame(rows, alternative_count)
        # Prime the cache once so subsequent measurements are cache hits.
        _ = run_sampled_eet(state, utilities, sample_size, alternative_count)
        for _ in range(repeats):
            state.rng().end_step(STEP_NAME)
            state.rng().begin_step(STEP_NAME)
            t0 = time.perf_counter()
            _ = run_sampled_eet(state, utilities, sample_size, alternative_count)
            timings.append(time.perf_counter() - t0)
        state.rng().end_step(STEP_NAME)
        del utilities
        del state
        gc.collect()
    return timings


def timed_cache_writes(rows: int, alternative_count: int, sample_size: int, repeats: int):
    timings = []
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        for i in range(repeats):
            state, _choosers = make_rng(
                base_dir / f"write_{i}", cache_enabled=True, rows=rows
            )
            utilities = build_utility_frame(rows, alternative_count)
            t0 = time.perf_counter()
            _ = run_sampled_eet(state, utilities, sample_size, alternative_count)
            timings.append(time.perf_counter() - t0)
            state.rng().end_step(STEP_NAME)
            del utilities
            del state
            gc.collect()
    return timings


def timed_save_only(sample: np.ndarray, repeats: int):
    timings = []
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        for i in range(repeats):
            file_path = out_dir / f"save_{i}.npy"
            t0 = time.perf_counter()
            np.save(file_path, sample)
            timings.append(time.perf_counter() - t0)
    return timings


def timed_legacy_random(rows: int, sample_size: int, repeats: int):
    timings = []
    with tempfile.TemporaryDirectory() as tmpdir:
        state, choosers = make_rng(Path(tmpdir) / "legacy", cache_enabled=False, rows=rows)
        for _ in range(repeats):
            state.rng().end_step(STEP_NAME)
            state.rng().begin_step(STEP_NAME)
            t0 = time.perf_counter()
            _ = state.rng().random_for_df(choosers, n=sample_size)
            timings.append(time.perf_counter() - t0)
        state.rng().end_step(STEP_NAME)
        del state
        gc.collect()
    return timings


def summarize(values):
    return statistics.median(values), min(values), max(values)


def main():
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Compare sampled EET Gumbel draw time with and without the random cache."
    )
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--alternative-count", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    uncached_samples = timed_uncached_sampled(
        args.rows, args.alternative_count, args.sample_size, args.repeats
    )
    uncached_timings = [t for t, _ in uncached_samples]
    sample = uncached_samples[-1][1]

    write_samples = timed_cache_writes(
        args.rows, args.alternative_count, args.sample_size, args.repeats
    )
    write_timings = write_samples

    cache_load_timings = timed_cached_loads(
        args.rows, args.alternative_count, args.sample_size, args.repeats
    )
    save_timings = timed_save_only(sample, args.repeats)
    legacy_timings = timed_legacy_random(args.rows, args.sample_size, args.repeats)

    uncached_median, uncached_min, uncached_max = summarize(uncached_timings)
    write_median, write_min, write_max = summarize(write_timings)
    load_median, load_min, load_max = summarize(cache_load_timings)
    save_median, save_min, save_max = summarize(save_timings)
    legacy_median, legacy_min, legacy_max = summarize(legacy_timings)

    print(
        f"rows={args.rows} alternative_count={args.alternative_count} "
        f"sample_size={args.sample_size} repeats={args.repeats}"
    )
    print(f"uncached sampled-eet timings: {', '.join(f'{t:.6f}' for t in uncached_timings)}")
    print(f"cache write timings:  {', '.join(f'{t:.6f}' for t in write_timings)}")
    print(f"cache load timings:   {', '.join(f'{t:.6f}' for t in cache_load_timings)}")
    print(f"save-only timings:    {', '.join(f'{t:.6f}' for t in save_timings)}")
    print(f"legacy random_for_df timings: {', '.join(f'{t:.6f}' for t in legacy_timings)}")
    print(f"uncached median: {uncached_median:.6f} s (min {uncached_min:.6f}, max {uncached_max:.6f})")
    print(f"cache write median: {write_median:.6f} s (min {write_min:.6f}, max {write_max:.6f})")
    print(f"cache load median:  {load_median:.6f} s (min {load_min:.6f}, max {load_max:.6f})")
    print(f"save-only median:   {save_median:.6f} s (min {save_min:.6f}, max {save_max:.6f})")
    print(f"legacy random_for_df median: {legacy_median:.6f} s (min {legacy_min:.6f}, max {legacy_max:.6f})")
    print(f"write speedup vs uncached: {uncached_median / write_median:.2f}x")
    print(f"load speedup vs uncached:  {uncached_median / load_median:.2f}x")
    print(f"legacy speedup vs uncached sampled-eet: {uncached_median / legacy_median:.2f}x")


if __name__ == "__main__":
    main()