**Problem `benchmarking_script`:**

Please see [./cs336_systems/benchmark_grid.py](./cs336_systems/benchmark_grid.py) for
the end-to-end benchmarking of the forward and backward passes.

Running on a single H200 SXM node gives following
results [mlflow](https://mlflow.sutroplanet.com/#/compare-runs?runs=[%22d4002721711e4d068c96b198ea7248d6%22,%223721f86d926c49499237b94bc80f5f71%22,%22676ade9594d14b5083c809b2819c2f30%22,%22cde59887fc694de1ac7fac76b12f217e%22,%22da1852caa3504bcb9ee5de52d241d4c8%22]&experiments=[%22131%22]).
As we can see, as the model size increase, both forward pass and backward pass elapses increase. The standard deviation
of elapses is relatively small.

Without warmup steps, the initial steps have longer forward/backward pass elapses.

**Problem `nsys_profile`:**

The following results are from running forward/backward for an LM with `d_model=128 and context_len=128` using
script [./cs336_systems/nsys_profile.py](./cs336_systems/nsys_profile.py). The resulting trace is [./trace/nsys_profile.nsys-rep](./trace/nsys_profile.nsys-rep).
The total forward pass time is 440.243ms. It matches with the results from python `timeit` library.


![image](./images/nsys-profile.png)

The CUDA kernel takes the most cumulative GPU time is `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_tn_align1>(T1::Params)` during forward pass. It is invoked 60 times for every forward pass. For backward pass, it is the same kernel that takes the most of time.


I saw another contributor of kernels are `OpScalar` type, probably related to 


```text
NVTX Range	Style	PID	TID	NVTX Inst	Kern Inst	Total Time	Avg	Med	Min	Max	StdDev	Kernel Name
:forward	PushPop	2828803	2828803	1	146	363.520 μs	2.489 μs	2.592 μs	1.984 μs	2.688 μs	216 ns	void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
:forward	PushPop	2828803	2828803	1	72	137.888 μs	1.915 μs	1.824 μs	1.792 μs	2.144 μs	137 ns	void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
```

    