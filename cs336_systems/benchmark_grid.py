from cs336_systems.benchmark import BenchmarkConfig, run_benchmark

if __name__ == '__main__':
    configs = [
        BenchmarkConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
        BenchmarkConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
        BenchmarkConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
        BenchmarkConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
        BenchmarkConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    ]

    for config in configs:
        run_benchmark(config)
