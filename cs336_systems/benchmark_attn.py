from dataclasses import asdict, dataclass
import logging
import timeit
import statistics
import torch
import gc

from cs336_basics.model import CausalMultiHeadSelfAttention, RotaryEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

device = 'cuda'


def get_gpu_usage_in_gb():
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    to_gb = 1024 ** 3

    return allocated / to_gb, reserved / to_gb


def benchmark_attn(batch_size: int, d_model: int, seq_len: int, num_heads: int, context_len: int, rope_theta: float):
    logger.info(
        f"Benchmarking with config: batch_size={batch_size}, d_model={d_model}, seq_len={seq_len}, num_heads={num_heads}, context_len={context_len}")

    assert seq_len <= context_len, "Sequence length cannot be greater than context length."

    warmup_step, benchmark_step = 5, 100
    positional_encoder = RotaryEmbedding(
        context_length=context_len,
        dim=d_model,
        theta=rope_theta
    )

    attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, positional_encoder=positional_encoder).to(
        device)
    x = torch.randn(size=(batch_size, seq_len, d_model)).to(device)

    # Benchmarking forward pass
    with torch.cuda.nvtx.range(f"forward"):

        for _ in range(warmup_step):
            out = attn(x)

        torch.cuda.synchronize(device)
        forward_start_timer = timeit.default_timer()
        for idx in range(benchmark_step):
            with torch.cuda.nvtx.range(f"forward_{idx}"):
                out = attn(x)

        torch.cuda.synchronize(device)
        forward_end_timer = timeit.default_timer()
        allocated, reserved = get_gpu_usage_in_gb()
        logger.info(
            f"{benchmark_step} forward pass duration: {forward_end_timer - forward_start_timer}, allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")

    # Benchmarking backward pass
    loss = out.sum()  # dummy loss
    with torch.cuda.nvtx.range(f"backward"):
        for _ in range(warmup_step):
            attn.zero_grad()
            loss.backward(retain_graph=True)

        torch.cuda.synchronize(device)
        backward_start_timer = timeit.default_timer()

        for idx in range(benchmark_step):
            attn.zero_grad()
            with torch.cuda.nvtx.range(f"backward_{idx}"):
                loss.backward(retain_graph=True)

        torch.cuda.synchronize(device)
        backward_end_timer = timeit.default_timer()
        allocated, reserved = get_gpu_usage_in_gb()
        logger.info(
            f"{benchmark_step} backward pass duration: {backward_end_timer - backward_start_timer}, allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")


if __name__ == '__main__':
    batch_size, context_len, rope_theta = 8, 32768, 10000.0

    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    for d_model in d_models:
        for seq_len in seq_lens:
            benchmark_attn(batch_size, d_model, seq_len, num_heads=1, context_len=context_len, rope_theta=rope_theta)

            gc.collect()
            torch.cuda.empty_cache()
