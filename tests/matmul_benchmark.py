import torch, time, gc
import qllmt
import numpy as np
import argparse
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import qllmt.functional
import qllmt.kernels
import fp8

# from qllmt.nn.triton_int8_matmul import int8_matmul
# from qllmt.nn.switchback import int8_matmul_mixed_dequantize_bf16
from vllm import _custom_ops as vllm_ops

num_warmup_steps = 10
num_bench_steps = 100

@torch.no_grad()
def torch_benchmark(a, b):
    # warmup
    for i in range(num_warmup_steps):
        out = torch.matmul(a, b)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = torch.matmul(a, b)
    torch.cuda.synchronize()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps


@torch.no_grad()
def cutlass_int8_benchmark(a, b):
    scale_a = a.new_ones(1, dtype=torch.float32)
    scale_b = b.new_ones(1, dtype=torch.float32)
    b = b.T
    # warmup
    for i in range(num_warmup_steps):
        # out = int8_matmul_mixed_dequantize_bf16(a, b, a.new_ones(1), a.new_ones(1), bias=None)
        # out = int8_matmul(a, b, a.new_ones(1))
        # out = qllmt.linear_a8_w8_bfp32_ofp32(a, b, 1.0).half()
        out = vllm_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16)
        # out_qllmt = qllmt.linear_a8_w8_bfp32_ofp32(a, b, 1.0).to(torch.bfloat16)
        # print(torch.allclose(out, out_qllmt))
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        # out = int8_matmul_mixed_dequantize_bf16(a, b, a.new_ones(1), a.new_ones(1), bias=None)
        # out = int8_matmul(a, b, a.new_ones(1))
        # out = qllmt.linear_a8_w8_bfp32_ofp32(a, b, 1.0).half()
        out = vllm_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16)
    torch.cuda.synchronize()
    

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps

@torch.no_grad()
def cutlass_fp8_benchmark(a, b):
    # warmup
    for i in range(num_warmup_steps):
        out = fp8.fp8_matmul_fastAcc(a, b, 1.)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = fp8.fp8_matmul_fastAcc(a, b, 1.)
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps


if __name__ == '__main__':

    cutlass_int8_speedups = []
    cutlass_fp8_speedups = []
    # triton_speedups = []
    bfloat_speedups = []
    input_sizes = [(512*8, 4096), (512*8, 8192), (512*8, 11008), (512*8, 28672)]
    # input_sizes = [(512*16, 28672), (512*16, 28672 * 2), (512*32, 28672 * 2)]
    for size in input_sizes:
        in_dim = size[0]
        mid_dim = size[1]
        out_dim = size[1]
        # integer random matrix
        a = torch.randint(2, 10, (in_dim, mid_dim), device='cuda', dtype=torch.float16)
        b = torch.randint(3, 12, (out_dim, mid_dim), device='cuda', dtype=torch.float16)

        torch_times = []
        cutlass_int8_times = []
        cutlass_fp8_times = []
        # triton_times = []
        bfloat_times = []

        torch_times.append(torch_benchmark(a, b.t()))
        bfloat_times.append(torch_benchmark(a.to(torch.bfloat16), b.t().to(torch.bfloat16)))

        cutlass_int8_times.append(cutlass_int8_benchmark(a.to(torch.int8), b.t().to(torch.int8).contiguous()))
        cutlass_fp8_times.append(cutlass_fp8_benchmark(a.to(torch.float8_e4m3fn), b.t().to(torch.float8_e4m3fn).contiguous()))
    

        print(f"Matrix size: {in_dim}x{mid_dim}x{out_dim}")
        print(f"Torch time: {np.mean(torch_times):.3f} +- {1.96 * np.std(torch_times):.3f}ms")
        print(f"BFLOAT16 time: {np.mean(bfloat_times):.3f} +- {1.96 * np.std(bfloat_times):.3f}ms")
        print(f"INT8 (CUTLASS) time: {np.mean(cutlass_int8_times):.3f} +- {1.96 * np.std(cutlass_int8_times):.3f}ms")
        print(f"FP8 (CUTLASS) time: {np.mean(cutlass_fp8_times):.3f} +- {1.96 * np.std(cutlass_fp8_times):.3f}ms")
        print(f"Torch/CUTLASS (INT8): {np.mean(torch_times) / np.mean(cutlass_int8_times):.3f}x")
        print(f"Torch/CUTLASS (FP8): {np.mean(torch_times) / np.mean(cutlass_fp8_times):.3f}x")
        cutlass_int8_speedups.append(np.mean(torch_times) / np.mean(cutlass_int8_times))
        cutlass_fp8_speedups.append(np.mean(torch_times) / np.mean(cutlass_fp8_times))
        bfloat_speedups.append(np.mean(torch_times) / np.mean(bfloat_times))
        print('--------------------')
    
     # plot results
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(cutlass_int8_speedups, '-o', label='vLLM (INT8)')
    # ax.plot(cutlass_int8_speedups, '-o', label='CUTLASS (INT8)')
    ax.plot(cutlass_fp8_speedups, '-o', label='CUTLASS (FP8)')
    ax.plot(bfloat_speedups, '-o', label='BFLOAT16')
    x_labels = [f'{B}x{M}x{M}' for B, M in input_sizes]
    ax.set_xticks(range(len(input_sizes)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Speedup')
    ax.set_title(f'INT8/FP8 MatMul Speedup (over FP16 Torch) on {torch.cuda.get_device_name()}')
    ax.legend()
    ax.grid()
    plt.savefig(f'8bit_matmul_speedups_{torch.cuda.get_device_name()}.png')