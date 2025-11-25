from os import PathLike
import argparse

import torch
import torch.nn as nn
import time
from contextlib import nullcontext

from cs336_systems.playground.toy_model import LanguageModel, get_model_config


class ProfileMemory:
    """Context manager to profile GPU memory usage and dump to pickle file."""

    def __init__(self, dump_path: str | PathLike):
        self.dump_path = dump_path

    def __enter__(self):
        torch.cuda.memory._record_memory_history()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.memory._dump_snapshot(self.dump_path)
        torch.cuda.memory._record_memory_history(enabled=None)


def benchmark_model(model, inputs, targets, criterion, precision_context, inference_only=False, num_warmup=5,
                    num_iterations=20):
    """
    Benchmark forward and backward passes with the given precision context.

    Args:
        model: The model to benchmark
        inputs: Input tensor
        targets: Target tensor
        criterion: Loss function
        precision_context: Context manager for precision (autocast or nullcontext)
        inference_only: If True, only benchmark forward pass without backward pass
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations to benchmark

    Returns:
        tuple: (forward_time, backward_time) in milliseconds. backward_time is 0 if inference_only=True
    """
    # Warmup
    for _ in range(num_warmup):
        if inference_only:
            with torch.no_grad(), precision_context:
                logits = model(inputs)
                # Reshape for loss calculation
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            with precision_context:
                logits = model(inputs)
                # Reshape for loss calculation
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            model.zero_grad()

    # Synchronize before benchmarking
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark forward and backward passes
    forward_times = []
    backward_times = []

    for _ in range(num_iterations):
        if not inference_only:
            model.zero_grad()

        # Time forward pass
        start = time.perf_counter()
        if inference_only:
            # Use no_grad to prevent gradient computation and save memory
            with torch.no_grad(), precision_context:
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            with precision_context:
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_end = time.perf_counter()

        forward_times.append((forward_end - start) * 1000)  # Convert to milliseconds

        # Time backward pass (only if training mode)
        if not inference_only:
            backward_start = time.perf_counter()
            loss.backward()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_end = time.perf_counter()

            backward_times.append((backward_end - backward_start) * 1000)  # Convert to milliseconds
            model.zero_grad()

    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times) if backward_times else 0.0

    return avg_forward_time, avg_backward_time


def run_benchmarks(model_sizes=None, precision_settings=None, seq_lens=None, inference_only=False):
    """Run benchmarks for all model sizes and precision settings."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 32
    vocab_size = 50000
    criterion = nn.CrossEntropyLoss()

    results = {}

    for seq_length in seq_lens:
        print("=" * 80 + f"SEQ LEN {seq_length}" + "=" * 80)
        print("Language Model Benchmarking: Forward and Backward Pass Times")
        print("=" * 160)
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Sequence length: {seq_length}")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Number of iterations: 20 (after 5 warmup iterations)")
        print("=" * 80)
        print()

        for size in model_sizes:
            config = get_model_config(size)

            print(f"\n{'=' * 80}")
            print(f"Model Size: {size}")
            print(f"Configuration: {config}")
            print(f"{'=' * 80}")

            inputs = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
            targets = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
            results[size] = {}

            precision_context_map = {
                'FP32': nullcontext(),
                'BF16': torch.autocast(device_type='cuda', dtype=torch.bfloat16),
                'FP16': torch.autocast(device_type='cuda', dtype=torch.float16),
            }

            for precision in precision_settings:
                print(f"\n{precision} Precision:")
                print("-" * 80)

                mode_str = "inference" if inference_only else "training"
                memory_profile_context = ProfileMemory(
                    f"./trace/playground/benchmark_toy_model_{size}_{precision}_seq_{seq_length}_{mode_str}.pickle")
                precision_context = precision_context_map[precision]

                with memory_profile_context:
                    try:
                        model = LanguageModel(
                            d_model=config['d_model'],
                            d_ff=config['d_ff'],
                            num_layers=config['num_layers'],
                            num_heads=config['num_heads'],
                            vocab_size=vocab_size
                        ).to(device)
                        forward_time, backward_time = benchmark_model(
                            model, inputs, targets, criterion, precision_context, inference_only=inference_only
                        )
                        results[size][precision] = {
                            'forward': forward_time,
                            'backward': backward_time,
                            'total': forward_time + backward_time
                        }

                        print(f"  Forward pass:  {forward_time:.3f} ms")
                        print(f"  Backward pass: {backward_time:.3f} ms")
                        print(f"  Total:         {forward_time + backward_time:.3f} ms")

                    except Exception as e:
                        print(f"  Error benchmarking {precision}: {e}")
                        continue

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Print summary table
        print("\n\n" + "=" * 80)
        print("SUMMARY: Total Time (Forward + Backward) in milliseconds")
        print("=" * 80)

        # Header
        header = f"{'Model Size':<12}"
        for precision in precision_settings:
            header += f"{precision:>12}"
        print(header)
        print("-" * 80)

        # Data rows
        for size in model_sizes:
            row = f"{size:<12}"
            for precision in precision_settings:
                if size in results and precision in results[size]:
                    total_time = results[size][precision]['total']
                    row += f"{total_time:>12.3f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)

        # Print speedup analysis
        print("\n\n" + "=" * 80)
        print("SPEEDUP ANALYSIS: Mixed Precision vs FP32")
        print("=" * 80)

        header = f"{'Model Size':<12}{'BF16 Speedup':>15}{'FP16 Speedup':>15}"
        print(header)
        print("-" * 80)

        for size in model_sizes:
            row = f"{size:<12}"

            if size in results and 'FP32' in results[size]:
                fp32_time = results[size]['FP32']['total']

                # BF16 speedup
                if 'BF16' in results[size]:
                    bf16_time = results[size]['BF16']['total']
                    bf16_speedup = fp32_time / bf16_time
                    row += f"{bf16_speedup:>15.2f}x"
                else:
                    row += f"{'N/A':>15}"

                # FP16 speedup
                if 'FP16' in results[size]:
                    fp16_time = results[size]['FP16']['total']
                    fp16_speedup = fp32_time / fp16_time
                    row += f"{fp16_speedup:>15.2f}x"
                else:
                    row += f"{'N/A':>15}"
            else:
                row += f"{'N/A':>15}{'N/A':>15}"

            print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark language models with different sizes and precision settings')
    parser.add_argument('--model-sizes', nargs='+', default=['small', 'medium', 'large', 'xl', '2.7B'],
                        choices=['small', 'medium', 'large', 'xl', '2.7B'],
                        help='Model sizes to benchmark (default: all)')
    parser.add_argument('--precision-settings', nargs='+', default=['FP32', 'BF16', 'FP16'],
                        choices=['FP32', 'BF16', 'FP16'],
                        help='Precision settings to benchmark (default: all)')
    parser.add_argument('--seq-lens', nargs='+', default=[128, 256, 512],
                        help='Sequence lengths to benchmark (default: 128, 256, 512)')
    parser.add_argument('--inference-only', action='store_true',
                        help='Profile memory for inference only (forward pass without backward pass)')

    args = parser.parse_args()
    run_benchmarks(model_sizes=args.model_sizes, precision_settings=args.precision_settings,
                   seq_lens=args.seq_lens, inference_only=args.inference_only)
