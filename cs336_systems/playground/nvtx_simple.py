import time
import torch.cuda.nvtx as nvtx


def simple():
    with nvtx.range("loop"):
        for idx in range(5):
            with nvtx.range(f"loop_{idx}"):
                time.sleep(idx / 100)


if __name__ == '__main__':
    simple()
