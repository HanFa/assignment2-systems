"""
Benchmark native DDP (manual all-reduce) with nsys tracing.

Usage:
    # Basic run (uses all GPUs, XL model)
    python -m cs336_systems.playground.distributed.naive_ddp_benchmarking

    # With nsys profiling
    nsys profile -c cudaProfilerApi -o trace/playground/distributed/naive_ddp_benchmarking/trace python -m cs336_systems.playground.distributed.naive_ddp_benchmarking

    # Custom config
    python -m cs336_systems.playground.distributed.naive_ddp_benchmarking \
        --model-size xl --batch-size 4 --seq-length 512 --num-gpus 4
"""
import os
import argparse
import logging

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx

from cs336_systems.playground.toy_model import LanguageModel, get_model_config

logger = logging.getLogger(__name__)


def setup(rank, world_size, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    dist.destroy_process_group()


def native_ddp_worker(rank, world_size, config, num_warmup, num_iterations, batch_size, seq_length, vocab_size):
    # Configure logging in worker process (spawned processes don't inherit logging config)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # Override any existing config
    )

    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)

    model = LanguageModel(
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        vocab_size=vocab_size
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model config: {config}")
        logger.info(f"Number of parameters: {num_params:,}")
        logger.info(f"Batch size: {batch_size}, Seq length: {seq_length}")
        logger.info(f"Starting warmup ({num_warmup} iterations)...")

    # Warmup
    for i in range(num_warmup):
        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size

        optimizer.step()

        if rank == 0:
            logger.info(f"  Warmup {i + 1}/{num_warmup} | Loss: {loss.item():.4f}")

    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        logger.info(f"Starting benchmark ({num_iterations} iterations)...")

    # Start CUDA profiler for nsys capture
    torch.cuda.cudart().cudaProfilerStart()

    # Benchmark with nvtx markers for nsys profiling
    for i in range(num_iterations):
        nvtx.range_push(f"iteration_{i}")

        optimizer.zero_grad()

        nvtx.range_push("forward")
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        nvtx.range_pop()

        nvtx.range_push("backward")
        loss.backward()
        torch.cuda.synchronize()
        nvtx.range_pop()

        nvtx.range_push("all_reduce")
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        torch.cuda.synchronize()
        nvtx.range_pop()

        nvtx.range_push("optimizer_step")
        optimizer.step()
        torch.cuda.synchronize()
        nvtx.range_pop()

        nvtx.range_pop()  # iteration

        if rank == 0:
            logger.info(f"  Iteration {i + 1}/{num_iterations} | Loss: {loss.item():.4f}")

    # Stop CUDA profiler
    torch.cuda.cudart().cudaProfilerStop()

    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        logger.info("Benchmark complete.")

    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Benchmark native DDP with nsys tracing')
    parser.add_argument('--model-size', type=str, default='xl',
                        choices=['small', 'medium', 'large', 'xl', '2.7B'],
                        help='Model size to benchmark (default: xl)')
    parser.add_argument('--num-warmup', type=int, default=5,
                        help='Number of warmup iterations (default: 5)')
    parser.add_argument('--num-iterations', type=int, default=10,
                        help='Number of benchmark iterations (default: 10)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size per GPU (default: 4)')
    parser.add_argument('--seq-length', type=int, default=512,
                        help='Sequence length (default: 512)')
    parser.add_argument('--vocab-size', type=int, default=50000,
                        help='Vocabulary size (default: 50000)')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    config = get_model_config(args.model_size)
    num_gpus = args.num_gpus or torch.cuda.device_count()

    logger.info(f"Using {num_gpus} GPUs for native DDP benchmark")
    logger.info(f"Model size: {args.model_size}")

    mp.spawn(
        fn=native_ddp_worker,
        args=(
            num_gpus,
            config,
            args.num_warmup,
            args.num_iterations,
            args.batch_size,
            args.seq_length,
            args.vocab_size
        ),
        nprocs=num_gpus,
        join=True
    )


if __name__ == '__main__':
    main()
