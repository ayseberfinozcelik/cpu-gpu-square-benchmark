import time

import numpy as np


ARRAY_SIZE = 1_000_000
REPEATS = 10
SEED = 42


def main() -> None:
    # Create a float32 array with random values.
    rng = np.random.default_rng(seed=SEED)
    values = rng.random(ARRAY_SIZE, dtype=np.float32)
    input_checksum = float(values.mean())

    # Warm-up run to reduce one-time overhead effects.
    _ = values * values

    elapsed_times = []
    squared_values = None
    for _ in range(REPEATS):
        start_time = time.perf_counter()
        squared_values = values * values
        elapsed_times.append(time.perf_counter() - start_time)

    average_seconds = sum(elapsed_times) / REPEATS
    best_seconds = min(elapsed_times)

    # Print a tiny checksum so the result is definitely used.
    assert squared_values is not None
    checksum = float(squared_values.mean())
    print(f"ARRAY_SIZE={ARRAY_SIZE}")
    print(f"SEED={SEED}")
    print(f"REPEATS={REPEATS}")
    print(f"INPUT_CHECKSUM_MEAN={input_checksum:.9f}")
    print(f"CPU_TIME_SECONDS_AVG={average_seconds:.9f}")
    print(f"CPU_TIME_SECONDS_BEST={best_seconds:.9f}")
    print(f"CPU_CHECKSUM_MEAN={checksum:.9f}")


if __name__ == "__main__":
    main()
