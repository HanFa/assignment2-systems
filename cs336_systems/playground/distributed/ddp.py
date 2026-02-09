"""
Benchmark communication-overlapped DDP (using DDPWrapper) with nsys tracing.

Usage:
    # Basic run (uses all GPUs, XL model)
    python -m cs336_systems.playground.distributed.ddp

    # With nsys profiling
    nsys profile -c cudaProfilerApi -o trace/playground/distributed/ddp/trace python -m cs336_systems.playground.distributed.ddp

    # Custom config
    python -m cs336_systems.playground.distributed.ddp \
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


class DDPWrapper(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.all_reduce_handles: list = []

        for param in self.module.parameters():
            with torch.no_grad():
                dist.broadcast(param, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._post_accumulate_grad_hook)

    def _post_accumulate_grad_hook(self, param: torch.nn.Parameter):
        grad = param.grad.data
        handle = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
        self.all_reduce_handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.all_reduce_handles:
            handle.wait()
        self.all_reduce_handles.clear()


def train_step(ddp_model, optimizer, criterion, inputs, targets, trace=False):
    optimizer.zero_grad()

    if trace:
        nvtx.range_push("forward")
    logits = ddp_model(inputs)
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    if trace:
        nvtx.range_pop()

    if trace:
        nvtx.range_push("backward+all_reduce")
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    if trace:
        torch.cuda.synchronize()
        nvtx.range_pop()

    if trace:
        nvtx.range_push("optimizer_step")
    optimizer.step()
    if trace:
        torch.cuda.synchronize()
        nvtx.range_pop()

    return loss


def ddp_worker(rank, world_size, config, num_warmup, num_iterations, batch_size, seq_length, vocab_size):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
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

    ddp_model = DDPWrapper(model)
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
        loss = train_step(ddp_model, optimizer, criterion, inputs, targets)

        if rank == 0:
            logger.info(f"  Warmup {i + 1}/{num_warmup} | Loss: {loss.item():.4f}")

    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        logger.info(f"Starting benchmark ({num_iterations} iterations)...")

    torch.cuda.cudart().cudaProfilerStart()

    for i in range(num_iterations):
        nvtx.range_push(f"iteration_{i}")

        loss = train_step(ddp_model, optimizer, criterion, inputs, targets, trace=True)

        nvtx.range_pop()  # iteration

        if rank == 0:
            logger.info(f"  Iteration {i + 1}/{num_iterations} | Loss: {loss.item():.4f}")

    torch.cuda.cudart().cudaProfilerStop()

    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        logger.info("Benchmark complete.")

    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Benchmark communication-overlapped DDP with nsys tracing')
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

    logger.info(f"Using {num_gpus} GPUs for overlapped DDP benchmark")
    logger.info(f"Model size: {args.model_size}")

    mp.spawn(
        fn=ddp_worker,
        args=(
            num_gpus,
            config,
            args.num_warmup,
            args.num_iterations,
            args.batch_size,
            args.seq_length,
            args.vocab_size,
        ),
        nprocs=num_gpus,
        join=True
    )


if __name__ == '__main__':
    main()
