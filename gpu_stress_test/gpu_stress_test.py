"""
Usage: 
```
python gpu_stress_test.py max_runtime_in_seconds
```

`max_runtime_in_seconds` is optional and defaults to 300 seconds (5 minutes)
"""


import math
import socket
import sys
import time

import torch

# GPU_MEMORY_IN_GB = 40
MAX_RUNTIME = 5 * 60  # Run for 5 minutes

def get_gpu_memory_in_gb() -> float:
    """
    Retrieves the total GPU memory using Pytorch and returns it in gigabytes.
    
    Returns:
        float: Total GPU memory in gigabytes, rounded up to the nearest whole number.
    """
    free_mem, total_mem = torch.cuda.mem_get_info()
    
    gpu_memory_in_gb = total_mem / 1_000_000_000  # 1 GB = 10^9 bytes

    return math.ceil(gpu_memory_in_gb)

def run_load() -> str:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    # Get the array size for a square array that fills 1/4 of memory with 2 byte values
    GPU_MEMORY_IN_GB = get_gpu_memory_in_gb()
    arr_size = (((GPU_MEMORY_IN_GB / 4) * 10**9) / 2) ** (1 / 2)
    arr_size = int(math.ceil(arr_size))
    num_gpus = torch.cuda.device_count()
    if num_gpus != 8:
        return f"Found wrong number of GPUS: {num_gpus}"

    Ts = [
        torch.ones(arr_size, arr_size, dtype=torch.bfloat16, device=f"cuda:{gpu_num}") for gpu_num in range(num_gpus)
    ]
    results = [
        torch.zeros(arr_size, arr_size, dtype=torch.bfloat16, device=f"cuda:{gpu_num}") for gpu_num in range(num_gpus)
    ]
    from_others = [
        torch.zeros(arr_size, arr_size, dtype=torch.bfloat16, device=f"cuda:{gpu_num}") for gpu_num in range(num_gpus)
    ]

    torch.manual_seed(12345)

    start_time = time.time()
    curr_loop_num = 0
    while time.time() - start_time < MAX_RUNTIME:
        # Matrix multiply into result
        [torch.matmul(T, T, out=result) for T, result in zip(Ts, results)]

        # Move into gpu curr_loop_num away
        for i in range(num_gpus):
            other_gpu = (curr_loop_num % (num_gpus - 1) + i + 1) % num_gpus
            other = from_others[other_gpu]
            original = results[i]
            other[:] = original

        # Check values are correct
        checks = [(other == result).sum() == result.numel() for other, result in zip(from_others, results)]
        if not all([check.item() for check in checks]):
            return "Issue with GPUS, values don't match"

        curr_loop_num += 1

    if curr_loop_num < num_gpus:
        return f"Few loops seen, only {curr_loop_num}"

    return f"All okay for {curr_loop_num} loops"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MAX_RUNTIME = int(sys.argv[1])
    hostname = socket.gethostname()
    try:
        print(f"{hostname}: {run_load()}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"{hostname}: out of memory {e}")
    except Exception as e:
        print(f"{hostname}: {e}")
