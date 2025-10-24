import cs336_basics

from cs336_systems.benchmark import BenchmarkConfig, run_benchmark
from cs336_systems.annotated import annotated_scaled_dot_product_attention

if __name__ == '__main__':
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    configs = [
        BenchmarkConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    ]
    for config in configs:
        config.mlflow_experiment_name = "cs336-assignment2-nsys-profile"

        for context_len in [128]:
            config.max_seq_len = context_len
            config.warmup_steps = 1000  # disable mlflow logging by setting large warmup steps
            config.mlflow_run_suffix = f"d_model{config.d_model}ctx{config.max_seq_len}"
            run_benchmark(config)
