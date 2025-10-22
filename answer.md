**Problem `benchmarking_script`:**

Please see [./cs336_systems/benchmark_grid.py](./cs336_systems/benchmark_grid.py) for
the end-to-end benchmarking of the forward and backward passes.

Running on a single H200 SXM node gives following results [mlflow](https://mlflow.sutroplanet.com/#/compare-runs?runs=[%22d4002721711e4d068c96b198ea7248d6%22,%223721f86d926c49499237b94bc80f5f71%22,%22676ade9594d14b5083c809b2819c2f30%22,%22cde59887fc694de1ac7fac76b12f217e%22,%22da1852caa3504bcb9ee5de52d241d4c8%22]&experiments=[%22131%22]). As we can see, as the model size increase, both forward pass and backward pass elapses increase. The standard deviation of elapses is relatively small.

Without warmup steps, the initial steps have longer forward/backward pass elapses. 



