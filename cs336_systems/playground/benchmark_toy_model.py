import torch
import torch.nn as nn
import time
from contextlib import nullcontext


class TransformerBlock(nn.Module):
    """A single transformer block with multi-head attention and feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)

        return x


class LanguageModel(nn.Module):
    """A simple language model with multiple transformer blocks."""

    def __init__(self, d_model: int, d_ff: int, num_layers: int, num_heads: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads)
            for _ in range(num_layers)
        ])

        # Output layer
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # Embed input tokens
        x = self.embedding(input_ids)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits


def benchmark_model(model, inputs, targets, criterion, precision_context, num_warmup=5, num_iterations=20):
    """
    Benchmark forward and backward passes with the given precision context.

    Args:
        model: The model to benchmark
        inputs: Input tensor
        targets: Target tensor
        criterion: Loss function
        precision_context: Context manager for precision (autocast or nullcontext)
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations to benchmark

    Returns:
        tuple: (forward_time, backward_time) in milliseconds
    """
    # Warmup
    for _ in range(num_warmup):
        with precision_context:
            logits = model(inputs)
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        model.zero_grad()

    # Synchronize before benchmarking
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark forward pass
    forward_times = []
    for _ in range(num_iterations):
        model.zero_grad()

        start = time.perf_counter()
        with precision_context:
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        forward_times.append((end - start) * 1000)  # Convert to milliseconds

        # Clean up gradients
        loss.backward()
        model.zero_grad()

    # Benchmark backward pass
    backward_times = []
    for _ in range(num_iterations):
        model.zero_grad()

        with precision_context:
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        backward_times.append((end - start) * 1000)  # Convert to milliseconds

    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)

    return avg_forward_time, avg_backward_time


def get_model_config(size):
    """Get model configuration based on size."""
    configs = {
        'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
        'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
        'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
        'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
        '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
    }
    return configs[size]


def run_benchmarks():
    """Run benchmarks for all model sizes and precision settings."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cpu':
        print("Warning: Running on CPU. Benchmarks will be slow and mixed precision may not be supported.")

    model_sizes = ['small', 'medium', 'large', 'xl', '2.7B']
    precision_settings = ['FP32', 'BF16', 'FP16']

    batch_size = 32
    seq_length = 128
    vocab_size = 50000
    criterion = nn.CrossEntropyLoss()

    results = {}

    print("=" * 80)
    print("Language Model Benchmarking: Forward and Backward Pass Times")
    print("=" * 80)
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

        # Create model
        model = LanguageModel(
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            vocab_size=vocab_size
        ).to(device)

        # Create dummy data
        inputs = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

        results[size] = {}

        for precision in precision_settings:
            print(f"\n{precision} Precision:")
            print("-" * 80)

            # Set up precision context
            precision_context = None
            if precision == 'FP32':
                precision_context = nullcontext()
            elif precision == 'BF16':
                if device.type == 'cuda' and torch.cuda.is_bf16_supported():
                    precision_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                else:
                    print(f"  BF16 not supported on {device.type}, skipping...")
                    continue
            elif precision == 'FP16':
                if device.type == 'cuda':
                    precision_context = torch.autocast(device_type='cuda', dtype=torch.float16)
                else:
                    print(f"  FP16 autocast not supported on {device.type}, skipping...")
                    continue

            try:
                forward_time, backward_time = benchmark_model(
                    model, inputs, targets, criterion, precision_context
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

        # Clean up
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

    # Print observations
    print("\n\n" + "=" * 80)
    print("OBSERVATIONS AND TRENDS")
    print("=" * 80)
    print("""
1. PRECISION PERFORMANCE:
   - Mixed precision (BF16/FP16) typically provides speedups over FP32
   - BF16 offers similar performance to FP16 but with better numerical stability
   - The speedup magnitude depends on hardware support for tensor cores

2. MODEL SIZE SCALING:
   - Larger models generally show better speedups with mixed precision
   - This is because larger models have more compute-intensive operations
   - Tensor cores are better utilized with larger matrix operations

3. MEMORY BANDWIDTH:
   - Mixed precision reduces memory bandwidth requirements (2 bytes vs 4 bytes)
   - This becomes more significant as model size increases
   - Memory-bound operations benefit more from reduced precision

4. FORWARD VS BACKWARD:
   - Backward pass typically takes longer than forward pass
   - Mixed precision can accelerate both passes
   - The ratio may vary based on model architecture and operations

5. HARDWARE CONSIDERATIONS:
   - Modern GPUs (Ampere, Ada, Hopper) have dedicated tensor cores for FP16/BF16
   - Speedup is more pronounced on GPUs with tensor core support
   - CPU performance may not show significant benefits from mixed precision
    """)

    print("=" * 80)


if __name__ == '__main__':
    run_benchmarks()
