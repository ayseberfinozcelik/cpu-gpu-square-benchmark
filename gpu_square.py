import time

import numpy as np
import torch


ARRAY_SIZE = 1_000_000
REPEATS = 10
SEED = 42


def get_gpu_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("No supported GPU backend found (CUDA or MPS).")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    device = get_gpu_device()

    # Use the same NumPy RNG and seed as cpu_square.py so input values match exactly.
    rng = np.random.default_rng(seed=SEED)
    values_np = rng.random(ARRAY_SIZE, dtype=np.float32)
    input_checksum = float(values_np.mean())
    values_cpu = torch.from_numpy(values_np)
    values_gpu = values_cpu.to(device)

    # Warm-up for fair timing on first GPU operation.
    _ = values_gpu * values_gpu
    synchronize(device)

    elapsed_times = []
    squared_values = None
    for _ in range(REPEATS):
        start_time = time.perf_counter()
        squared_values = values_gpu * values_gpu
        synchronize(device)
        elapsed_times.append(time.perf_counter() - start_time)

    compute_average_seconds = sum(elapsed_times) / REPEATS
    compute_best_seconds = min(elapsed_times)

    # End-to-end timing (Host->Device transfer + compute + Device->Host transfer).
    end_to_end_times = []
    squared_values_cpu = None
    for _ in range(REPEATS):
        start_time = time.perf_counter()
        values_gpu_each = values_cpu.to(device)
        squared_values_gpu_each = values_gpu_each * values_gpu_each
        squared_values_cpu = squared_values_gpu_each.to("cpu")
        synchronize(device)
        end_to_end_times.append(time.perf_counter() - start_time)

    end_to_end_average_seconds = sum(end_to_end_times) / REPEATS
    end_to_end_best_seconds = min(end_to_end_times)

    assert squared_values is not None
    assert squared_values_cpu is not None
    checksum = float(squared_values.mean().item())
    checksum_end_to_end = float(squared_values_cpu.mean().item())
    print(f"ARRAY_SIZE={ARRAY_SIZE}")
    print(f"SEED={SEED}")
    print(f"REPEATS={REPEATS}")
    print(f"GPU_DEVICE={device}")
    print(f"INPUT_CHECKSUM_MEAN={input_checksum:.9f}")
    print(f"GPU_COMPUTE_TIME_SECONDS_AVG={compute_average_seconds:.9f}")
    print(f"GPU_COMPUTE_TIME_SECONDS_BEST={compute_best_seconds:.9f}")
    print(f"GPU_END_TO_END_TIME_SECONDS_AVG={end_to_end_average_seconds:.9f}")
    print(f"GPU_END_TO_END_TIME_SECONDS_BEST={end_to_end_best_seconds:.9f}")
    print(f"GPU_CHECKSUM_MEAN={checksum:.9f}")
    print(f"GPU_END_TO_END_CHECKSUM_MEAN={checksum_end_to_end:.9f}")


if __name__ == "__main__":
    main()
