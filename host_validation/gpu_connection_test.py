import argparse
import os
import resource
from typing import List
from typing import Optional
from typing import Tuple

import attr
import torch
import torch.distributed as dist
from loguru import logger
from utils.dist import get_local_rank
from utils.dist import init_dist
from utils.serialization import serialize_to_json
from utils.timer import Timer


@attr.s(auto_attribs=True, frozen=True)
class NvlinkResultsSchema:
    # Each result represents the duration of a single round of the nvlink test in ms
    results: Tuple[float, ...] = attr.ib(converter=tuple)


@attr.s(auto_attribs=True, frozen=True)
class IbResultsSchema:
    # Each result represents the duration of a single round of the ib test in ms
    results: Tuple[float, ...] = attr.ib(converter=tuple)
    rail: int


def run_checks(dim_items: int, loops: int, rail: Optional[int] = None) -> List[float]:
    """
    Runs some diagnostics to check for gpu and communication issues, either communicating over nvlink or ib
    """

    timer = Timer()

    if not rail:
        device_id = get_local_rank()
    else:
        device_id = rail

    with timer("cuda"):
        device = torch.device("cuda", device_id)
        torch.cuda.set_device(device_id)
        buffer = torch.ones((dim_items, dim_items), device=device, dtype=torch.float64)

    # warmup
    dist.all_reduce(buffer, op=dist.ReduceOp.AVG, async_op=False)

    results = []
    for i in range(loops):
        with timer(f"all_reduce_{i}"):
            with timer("send"):
                waiter = dist.all_reduce(buffer, op=dist.ReduceOp.AVG, async_op=True)
            with timer("sync"):
                waiter.wait()
                dist.barrier()
            with timer("stat"):
                buffer_sum = buffer.sum().item()

        results.append(timer[f"all_reduce_{i}"])
    return results


def run_ib(
    master_addr: str, master_port: int, rank: int, world_size: int, rail: int, dims: int, loops: int, force_ib: bool
) -> None:
    if force_ib:
        os.environ.update(
            {
                "NCCL_P2P_DISABLE": "1",
                "NCCL_SHM_DISABLE": "1",
            }
        )
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{master_addr}:{master_port}", rank=rank, world_size=world_size
    )
    logger.info(f"inited dist for ib_test: rank {rank}, world size {world_size}, rail {rail}")

    results = run_checks(dim_items=int(dims**0.5), rail=rail, loops=loops)

    logger.info(f"completed ib_test: rank {rank}, world size {world_size}, rail {rail}: result {results}")

    # this is for the parent to grab in stdout
    results_schema = IbResultsSchema(results=tuple(results), rail=rail)
    # Can't use a block here since it can interleave badly with the other rails
    print(serialize_to_json(results_schema))


def run_nvlink(dims: int, loops: int) -> None:
    logger.info(f"starting nvlink_test")
    init_dist()
    rank = get_local_rank()
    logger.info(f"inited dist for nvlink_test with rank {rank}")

    results = run_checks(dim_items=int(dims**0.5), loops=loops)

    logger.info(f"completed nvlink_test: result {results}")

    # this is for the parent to grab in stdout
    # a block could probably be used here instead, but just leave it consistent with the above test
    results_schema = NvlinkResultsSchema(results=tuple(results))
    print(serialize_to_json(results_schema))


@logger.catch(reraise=True)
def main() -> None:
    parser = argparse.ArgumentParser()
    command_parsers = parser.add_subparsers(dest="command")

    nvlink_parser = command_parsers.add_parser(
        "nvlink",
        description="Run work to exercise the nvlink and time the communication, example usage: 'gpu_connection_test.py nvlink --dims 10000 --loops 10'",
    )
    nvlink_parser.add_argument(
        "--dims",
        type=int,
        default=1_000_000_000,
        help="items in array to all_reduce, specifically we create a sqrt(dim_items) x sqrt(dim_items) array",
    )
    nvlink_parser.add_argument("--loops", type=int, default=200, help="number of loops of allreduce to run")
    nvlink_parser.set_defaults(func=run_nvlink)

    ib_parser = command_parsers.add_parser(
        "ib",
        description="Run work to exercise the infiniband and time the communication, example usage: 'gpu_connection_test ib --rail 0 --master_addr 10.0.200.1 --master_port 5001 --rank 0 --world_size 8', the master_addr and master_port should be the same for all gpus in a group",
    )
    ib_parser.add_argument("--rail", type=int, required=True, help="the rail being run on")
    ib_parser.add_argument("--master_addr", required=True)
    ib_parser.add_argument("--master_port", type=int, required=True)
    ib_parser.add_argument("--rank", type=int, required=True, help="the rank of the machine running this process")
    ib_parser.add_argument("--world_size", type=int, required=True, help="the total number of gpus")
    ib_parser.add_argument(
        "--dims",
        type=int,
        default=1_000_000_000,
        help="items in array to all_reduce, specifically we create a sqrt(dim_items) x sqrt(dim_items) array",
    )
    ib_parser.add_argument("--loops", type=int, default=200, help="number of loops of allreduce to run")
    ib_parser.add_argument("--force_ib", action="store_true", help="whether to force communication over infiniband")
    ib_parser.set_defaults(func=run_ib)

    r_soft, r_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (r_hard, r_hard))
    args = parser.parse_args()
    args_to_ignore = ("suppress_errors", "func", "command")
    args.func(**{k: v for k, v in vars(args).items() if k not in args_to_ignore})


if __name__ == "__main__":
    main()
