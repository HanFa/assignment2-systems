from dataclasses import asdict, dataclass
import logging
import timeit
import statistics
import torch
import torch.nn as nn
import mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from cs336_basics.model import BasicsTransformerLM


@dataclass
class BenchmarkConfig:
    # Model configurations
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0
    max_seq_len: int = 1024
    vocab_size: int = 10000
    num_layers: int = 12

    # Data loader configurations
    batch_size = 4

    # Optimizer configurations
    learning_rate: float = 1e-4

    # Training loop configurations
    warmup_steps: int = 5
    max_steps: int = 10
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging infra
    mlflow_tracking_uri: str = "http://mlflow.sutroplanet.com"
    mlflow_experiment_name: str = "cs336-assignment2-benchmarking"
    mlflow_run_suffix: str | None = None
    enable_logging: bool = False

    def __str__(self):
        prefix = f"d_model{self.d_model}num_layer{self.num_layers}"
        suffix = "" if self.mlflow_run_suffix is None else self.mlflow_run_suffix
        return f"{prefix}{suffix}"


def generate_rand_batch(cfg: BenchmarkConfig) -> tuple[torch.Tensor, torch.Tensor]:
    input_data = torch.randint(0, cfg.vocab_size, size=(cfg.batch_size, cfg.max_seq_len), dtype=torch.long,
                               device=cfg.device)
    targets = torch.roll(input_data, shifts=-1, dims=1)
    targets[:, -1] = 0
    return input_data, targets


def run_benchmark(cfg: BenchmarkConfig):
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    with mlflow.start_run(run_name=str(cfg)):
        mlflow.log_params(asdict(cfg))

        with torch.cuda.nvtx.range(f"load_model"):
            lm = BasicsTransformerLM(
                vocab_size=cfg.vocab_size,
                context_length=cfg.max_seq_len,
                d_model=cfg.d_model,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                d_ff=cfg.d_ff,
                rope_theta=cfg.rope_theta
            ).to(cfg.device)

        with torch.cuda.nvtx.range(f"generate_input_batch"):
            input_data, targets = generate_rand_batch(cfg)

        optimizer = torch.optim.AdamW(lm.parameters(), lr=cfg.learning_rate)

        forward_times = []
        backward_times = []

        for step in range(cfg.max_steps):
            with torch.cuda.nvtx.range(f"step_{step}"):
                timer_forward_start = timeit.default_timer()
                with torch.cuda.nvtx.range(f"forward"):
                    logits = lm(input_data)
                    torch.cuda.synchronize(cfg.device)
                timer_forward_end = timeit.default_timer()
                forward_elapsed = timer_forward_end - timer_forward_start

                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    targets.reshape(-1)
                )

                timer_backward_start = timeit.default_timer()
                with torch.cuda.nvtx.range(f"backward"):
                    loss.backward()
                    torch.cuda.synchronize(cfg.device)
                timer_backward_end = timeit.default_timer()
                backward_elapsed = timer_backward_end - timer_backward_start

                with torch.cuda.nvtx.range(f"optimizer_step"):
                    optimizer.step()

            if step < cfg.warmup_steps:
                continue

            forward_times.append(forward_elapsed)
            backward_times.append(backward_elapsed)

            mlflow.log_metric("loss", loss.item(), step=step)
            mlflow.log_metric("forward_time", forward_elapsed, step=step)
            mlflow.log_metric("backward_time", backward_elapsed, step=step)

            if step % 1 == 0:
                avg_forward = sum(forward_times) / len(forward_times)
                avg_backward = sum(backward_times) / len(backward_times)

                std_forward = statistics.stdev(forward_times) if len(forward_times) > 1 else 0
                std_backward = statistics.stdev(backward_times) if len(backward_times) > 1 else 0

                mlflow.log_metric("avg_forward_time", avg_forward, step=step)
                mlflow.log_metric("avg_backward_time", avg_backward, step=step)
                mlflow.log_metric("std_forward_time", std_forward, step=step)
                mlflow.log_metric("std_backward_time", std_backward, step=step)

                logger.info(
                    f"Step {step}: Loss = {loss.item():.4f}, Forward: {forward_elapsed:.4f}s (avg: {avg_forward:.4f}s,"
                    f" std: {std_forward:.4f}s), Backward: {backward_elapsed:.4f}s (avg: {avg_backward:.4f}s,"
                    f" std: {std_backward:.4f}s)")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    cfg = BenchmarkConfig()

    with torch.cuda.nvtx.range("run_benchmark"):
        run_benchmark(cfg)
