# HW4 - CPU vs GPU Square Benchmark (Apple M4)

## Task Summary
1. Create a float array of size `1,000,000`
2. Fill it with random values
3. Compute the square of each value
4. Measure runtime on CPU and GPU

## Hardware Discovery (How It Was Verified)
Commands used:
```bash
system_profiler SPHardwareDataType
system_profiler SPDisplaysDataType
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.backends.mps.is_available())"
```

Detected hardware:
- CPU Type: Apple M4 (10 cores: 4 Performance + 6 Efficiency)
- GPU Type: Apple M4 integrated GPU (10 cores, Metal supported)
- PyTorch backend check: `CUDA available: False`, `MPS available: True`

## Deliverable Files
- `cpu_square.py`: CPU implementation (NumPy)
- `gpu_square.py`: GPU implementation (PyTorch on Apple `mps`)

## Why CUDA Is Not Used
This computer has an Apple M4 GPU, not an NVIDIA GPU.  
CUDA requires NVIDIA hardware and CUDA drivers/toolkit.  
On Apple Silicon, the native GPU path is Metal/MPS, so GPU execution is done with PyTorch `mps`.

## Input Consistency Between CPU and GPU
Both scripts use:
- Same RNG library: NumPy `default_rng`
- Same seed: `42`
- Same dtype: `float32`
- Same array size: `1,000,000`

Validation:
- `INPUT_CHECKSUM_MEAN` is identical in both runs: `0.499684811`

## Timing Methodology
- CPU timing: only the square computation loop
- GPU compute timing: square kernel on preloaded GPU tensor (`compute-only`)
- GPU end-to-end timing: Host->Device transfer + compute + Device->Host transfer

## Measured Results
- Repeats: `10`
- CPU average time: `0.000086663` s
- CPU best time: `0.000065625` s
- GPU compute-only average time: `0.000491209` s
- GPU compute-only best time: `0.000349959` s
- GPU end-to-end average time: `0.001233950` s
- GPU end-to-end best time: `0.000768042` s
- CPU output checksum mean: `0.332950830`
- GPU output checksum mean: `0.332950801`

## Comment on the Result
For this workload, CPU is faster than GPU on this machine.  
The operation is very simple, so GPU launch/synchronization and transfer overhead dominate runtime.  
The CPU/GPU output checksums are almost identical; the tiny difference is expected from floating-point accumulation order differences.

## Run Instructions
```bash
python3 -m pip install numpy torch
python3 cpu_square.py
python3 gpu_square.py
```
