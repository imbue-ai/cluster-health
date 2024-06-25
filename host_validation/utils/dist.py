import datetime
import os

import torch.distributed as dist

# Have collectives timeout after 10 minutes instead of the default 30 minutes.
DIST_TIMEOUT = datetime.timedelta(minutes=10)


def init_dist() -> None:
    """Initialize distributed process group."""
    if "RANK" in os.environ:
        # defaults to initializing from environment variables
        dist.init_process_group(backend="nccl", timeout=DIST_TIMEOUT)
    else:
        # this is a dummy singlegpu setup
        dist.init_process_group(
            backend="nccl", world_size=1, rank=0, init_method="tcp://localhost:12345", timeout=DIST_TIMEOUT
        )


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))
