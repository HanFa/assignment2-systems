import os
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    backend: str  # "gloo" or "nccl"
    device: str  # "cpu" or "cuda"
    data_size: str  # "1MB", "10MB", "100MB", "1GB"
    num_processes: int  # 2, 4, or 6
    warmup: int


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    duration: float  # seconds, measured from rank 0


@dataclass
class BenchmarkResults:
    results: List[BenchmarkResult] = field(default_factory=list)

    def add(self, config: BenchmarkConfig, duration: float):
        self.results.append(BenchmarkResult(config=config, duration=duration))

    def group_by_backend(self) -> dict[str, List[BenchmarkResult]]:
        groups: dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            label = f"{r.config.backend}+{r.config.device}"
            groups.setdefault(label, []).append(r)
        return groups


def run_benchmark(rank: int, config: BenchmarkConfig, results_dict=None):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(config.backend, rank=rank, world_size=config.num_processes)

    if config.device == 'cuda':
        torch.cuda.set_device(rank)

    data_size_map = {
        '1MB': 1e6 / 4,
        '10MB': 1e7 / 4,
        '100MB': 1e8 / 4,
        '1GB': 1e9 / 4
    }

    data_num = int(data_size_map[config.data_size])
    device = f'cuda:{rank}' if config.device == 'cuda' else config.device
    data = torch.randint(0, 10, size=(data_num,), dtype=torch.float32, device=device)

    # Warmup iterations
    for _ in range(config.warmup):
        dist.all_reduce(data, async_op=False)

    # Actual benchmark
    def timed_func():
        dist.all_reduce(data, async_op=False)
        if config.device == 'cuda':
            torch.cuda.synchronize(device)

    duration = timeit.timeit(timed_func, number=1)
    print(f"rank {rank}: all_reduce duration = {duration:.6f}s")

    # Store rank 0 result for plotting
    if rank == 0 and results_dict is not None:
        key = (config.backend, config.device, config.data_size, config.num_processes)
        results_dict[key] = duration

    dist.destroy_process_group()


def make_plots(results: BenchmarkResults):
    data_sizes = ['1MB', '10MB', '100MB', '1GB']
    num_processes_list = [2, 4, 6]

    # Build lookup: backend_label -> (data_size, num_procs) -> duration
    groups = results.group_by_backend()
    backends: dict[str, dict[tuple[str, int], float]] = {}
    for label, group in groups.items():
        backends[label] = {
            (r.config.data_size, r.config.num_processes): r.duration for r in group
        }

    # Plot 1: Duration vs data size, one subplot per backend, lines per num_processes
    fig, axes = plt.subplots(1, len(backends), figsize=(7 * len(backends), 5), squeeze=False)
    for idx, (label, data) in enumerate(sorted(backends.items())):
        ax = axes[0][idx]
        for np in num_processes_list:
            durations = [data.get((ds, np), float('nan')) for ds in data_sizes]
            ax.plot(data_sizes, durations, marker='o', label=f'{np} procs')
        ax.set_title(f'All-Reduce: {label}')
        ax.set_xlabel('Data Size')
        ax.set_ylabel('Duration (s)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_reduce_by_data_size.png', dpi=150)
    print("Saved all_reduce_by_data_size.png")
    plt.close()

    # Plot 2: Duration vs num_processes, one subplot per backend, lines per data_size
    fig, axes = plt.subplots(1, len(backends), figsize=(7 * len(backends), 5), squeeze=False)
    for idx, (label, data) in enumerate(sorted(backends.items())):
        ax = axes[0][idx]
        for ds in data_sizes:
            durations = [data.get((ds, np), float('nan')) for np in num_processes_list]
            ax.plot(num_processes_list, durations, marker='o', label=ds)
        ax.set_title(f'All-Reduce: {label}')
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Duration (s)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_reduce_by_num_procs.png', dpi=150)
    print("Saved all_reduce_by_num_procs.png")
    plt.close()

    # Plot 3: Bar chart comparing backends for each data size (fixed num_procs)
    if len(backends) > 1:
        fig, axes = plt.subplots(1, len(num_processes_list), figsize=(6 * len(num_processes_list), 5), squeeze=False)
        backend_labels = sorted(backends.keys())
        x = range(len(data_sizes))
        width = 0.8 / len(backend_labels)
        for pidx, np in enumerate(num_processes_list):
            ax = axes[0][pidx]
            for bidx, bl in enumerate(backend_labels):
                durations = [backends[bl].get((ds, np), float('nan')) for ds in data_sizes]
                offset = (bidx - len(backend_labels) / 2 + 0.5) * width
                ax.bar([i + offset for i in x], durations, width=width, label=bl)
            ax.set_title(f'{np} Processes')
            ax.set_xlabel('Data Size')
            ax.set_ylabel('Duration (s)')
            ax.set_xticks(list(x))
            ax.set_xticklabels(data_sizes)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('all_reduce_backend_comparison.png', dpi=150)
        print("Saved all_reduce_backend_comparison.png")
        plt.close()


if __name__ == '__main__':
    backends_devices = [('gloo', 'cpu'), ('nccl', 'cuda')]
    data_sizes = ['1MB', '10MB', '100MB', '1GB']
    num_processes_list = [2, 4, 6]

    manager = mp.Manager()
    results_dict = manager.dict()

    for backend, device in backends_devices:
        for data_size in data_sizes:
            for num_processes in num_processes_list:
                config = BenchmarkConfig(
                    backend=backend,
                    device=device,
                    data_size=data_size,
                    num_processes=num_processes,
                    warmup=5
                )
                print(f"\n=== Benchmark: {backend}+{device}, {data_size}, {num_processes} processes ===")
                mp.spawn(fn=run_benchmark, args=(config, results_dict), nprocs=config.num_processes, join=True)

    # Convert to structured results and plot
    results = BenchmarkResults()
    for (backend, device, data_size, num_processes), duration in results_dict.items():
        config = BenchmarkConfig(
            backend=backend, device=device, data_size=data_size,
            num_processes=num_processes, warmup=5,
        )
        results.add(config, duration)
    print(f"\nCollected {len(results.results)} benchmark results")
    make_plots(results)
