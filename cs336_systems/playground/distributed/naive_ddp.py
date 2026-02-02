import os
import logging

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

from cs336_systems.playground.toy_model import ToyModel


def setup(rank, word_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("gloo", rank=rank, world_size=word_size)


def cleanup():
    dist.destroy_process_group()


def native_ddp_worker(rank, world_size, return_dict):
    setup(rank, world_size)

    torch.manual_seed(42)

    model = ToyModel(20, 5)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data = torch.rand(8, 20)
    target = torch.rand(8, 5)

    logger.info(f"[Rank {rank}] Starting training...")

    for epoch in range(5):
        optimizer.zero_grad()

        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size

        optimizer.step()

        logger.info(f"[Rank {rank}] Epoch {epoch} | Loss: {loss.item():.4f}")

    if rank == 0:
        return_dict["ddp_state_dict"] = {k: v.clone() for k, v in model.state_dict().items()}

    cleanup()


def train_without_ddp():
    torch.manual_seed(42)

    model = ToyModel(20, 5)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data = torch.rand(8, 20)
    target = torch.rand(8, 5)

    for epoch in range(5):
        optimizer.zero_grad()

        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        optimizer.step()

        logger.info(f"[No DDP] Epoch {epoch} | Loss: {loss.item():.4f}")

    return model


def compare_models(ddp_state_dict, no_ddp_state_dict):
    logger.info("\n=== Model Comparison ===")
    all_close = True
    for key in ddp_state_dict:
        ddp_param = ddp_state_dict[key]
        no_ddp_param = no_ddp_state_dict[key]
        is_close = torch.allclose(ddp_param, no_ddp_param, atol=1e-6)
        max_diff = (ddp_param - no_ddp_param).abs().max().item()
        logger.info(f"{key}: match={is_close}, max_diff={max_diff:.6e}")
        if not is_close:
            all_close = False
    logger.info(f"\nAll parameters match: {all_close}")
    return all_close


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    num_proc = 4

    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(fn=native_ddp_worker, args=(num_proc, return_dict), nprocs=num_proc, join=True)

    ddp_state_dict = return_dict["ddp_state_dict"]

    no_ddp_model = train_without_ddp()
    no_ddp_state_dict = no_ddp_model.state_dict()

    compare_models(ddp_state_dict, no_ddp_state_dict)
