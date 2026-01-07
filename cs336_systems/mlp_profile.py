from typing import Callable

import torch
import torch.nn as nn
from torch._C._profiler import ProfilerActivity


def get_device():
    return 'cuda'


class MLP(nn.Module):

    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = nn.functional.gelu(x)

        return x


def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda: operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda: operation(x, y)


def profile(description: str, run: Callable, num_wramups: int = 1, with_stack: bool = False):
    for _ in range(num_wramups):
        run()

    if torch.cuda.is_available():
        torch.cuda.synchronize(get_device())

    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=with_stack,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        run()
        torch.cuda.synchronize(get_device())

    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")

    return table


if __name__ == '__main__':
    add_function = lambda a, b: a + b
    add_profile = profile("add", run_operation2(dim=2048, operation=add_function))

    print("*" * 10 + "add_profile" + "*" * 10)
    print(add_profile)


    matmul_function = lambda a, b: a @ b
    matmul_profile = profile("matmul", run_operation2(dim=2048, operation=matmul_function))
    print("*" * 10 + "matmul_profile_2048" + "*" * 10)
    print(matmul_profile)

    matmul_function = lambda a, b: a @ b
    matmul_profile = profile("matmul", run_operation2(dim=128, operation=matmul_function))
    print("*" * 10 + "matmul_profile_128" + "*" * 10)
    print(matmul_profile)