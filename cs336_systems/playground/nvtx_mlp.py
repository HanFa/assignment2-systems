import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from typing import Callable


def get_device():
    return 'cuda'


class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""

    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for idx, layer in enumerate(self.layers):
            with nvtx.range(f"forward_layer_{idx}"):
                x = layer(x)
                x = torch.nn.functional.gelu(x)
        return x


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> None:
    with nvtx.range("run_mlp"):
        with nvtx.range("define_model"):
            # Define a model (with random weights)
            model = MLP(dim, num_layers).to(get_device())

        optimizer = torch.optim.AdamW(model.parameters())

        with nvtx.range("define_input"):
            # Define an input (random)
            x = torch.randn(batch_size, dim, device=get_device())

        with nvtx.range("optimizer_zero_grad"):
            optimizer.zero_grad()

        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            with nvtx.range("forward_pass"):
                # Forward
                y = model(x).mean()

            with nvtx.range("backward_pass"):
                # Backward
                y.backward()

            with nvtx.range("optimizer_step"):
                optimizer.step()


if __name__ == '__main__':
    run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=5)
