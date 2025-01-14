import torch
import qllmt
import numpy as np
import time
import matplotlib.pyplot as plt
import fast_hadamard_transform as fht
from math import ceil, log2
import fast_hadamard_transform

num_warmup_steps = 3
num_bench_steps = 10

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)



def had(x):
    had_K, K = qllmt.functional.hadamard.get_hadK(x.shape[-1], device=x.device, dtype=x.dtype)
    if K != 1:
        return qllmt.functional.hadamard.matmul_hadU_cuda(x, had_K.to(x.device), K)
    else:
        return qllmt.functional.hadamard.matmul_hadU_cuda(x, had_K, K)

def had_with_padding(x):
    d1, d2 = x.shape
    if not is_pow2(d2):
        d2_padded = 2 ** ceil(log2(d2))
        # x = torch.cat([x, x.new_zeros(d1, d2_padded - d2)], dim=1)
        x_padded = x.new_empty(d1, d2_padded)
        x_padded[:, :d2] = x
        x_padded[:, d2:] = 0
        x = x_padded
    return fht.hadamard_transform(x, 1.0/torch.tensor(x.shape[-1]).sqrt())

slice_had_points = {
    6144: [0, 4096, 6144],
    11008: [0, 8192, 10240, 10752, 11008],
    13824: [0, 8192, 12288, 13312, 13824],
}

def had_with_slicing(X):
    d1, d2 = X.shape
    if is_pow2(d2):
        return fht.hadamard_transform(X, 1.0/torch.tensor(X.shape[-1]).sqrt())

    points = slice_had_points[d2]
    XH = torch.empty_like(X)
    for i, (p1, p2) in enumerate(zip(points[:-1], points[1:])):
        XH[:, p1:p2] = fht.hadamard_transform(X[:, p1:p2], 1.0/torch.tensor(p2-p1).sqrt())
    return XH

def had_transpose(x):
    hadK, K = qllmt.functional.hadamard.get_hadK(x.shape[-2], transpose=True, device=x.device, dtype=x.dtype)
    return qllmt.functional.hadamard.matmul_hadU_cuda(x.transpose(-1, -2).contiguous(),
                                hadK, K).transpose(-1, -2).contiguous()

def had_group(x):
    n = x.shape[-1]

    if is_pow2(n):
        return fast_hadamard_transform.hadamard_transform(x, 1.0/torch.tensor(n).sqrt()) 
    else:
        if n == 11008:
            K = 43
        elif n == 13824:
            K = 27
        elif n == 6144:
            K = 3
        else:
            raise ValueError(f"Unsupported size {n}")

        return fast_hadamard_transform.hadamard_transform(x.view(-1, K, n // K).contiguous(),
                                                           1.0/torch.tensor(n // K).sqrt()).reshape(x.shape)

@torch.no_grad()
def triton_quant_benchmark(a):
    # warmup
    for i in range(num_warmup_steps):
        out = qllmt.functional.quantization.per_tensor_int8_quant_triton(a)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = qllmt.functional.quantization.per_tensor_int8_quant_triton(a)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps


@torch.no_grad()
def had_benchmark(a):
    # warmup
    for i in range(num_warmup_steps):
        out = had(a)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = had(a)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps

@torch.no_grad()
def had_with_padding_benchmark(a):
    # warmup
    for i in range(num_warmup_steps):
        out = had_with_padding(a)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = had_with_padding(a)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps

@torch.no_grad()
def had_with_slicing_benchmark(a):
    # warmup
    for i in range(num_warmup_steps):
        out = had_with_slicing(a)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = had_with_slicing(a)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps

@torch.no_grad()
def group_had_benchmark(a):
    # warmup
    for i in range(num_warmup_steps):
        out = had_group(a)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = had_group(a)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps


@torch.no_grad()
def had_transpose_benchmark(a):
    # warmup
    for i in range(num_warmup_steps):
        out = had_transpose(a)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = had_transpose(a)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps



@torch.no_grad()
def cutlass_benchmark(a, b):
    # warmup
    for i in range(num_warmup_steps):
        out = qllmt.linear_a8_w8_bfp32_ofp32(a, b, 1.0).half()
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = qllmt.linear_a8_w8_bfp32_ofp32(a, b, 1.0).half()
    torch.cuda.synchronize()
    

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps


@torch.no_grad()
def torch_matmul_benchmark(a, b):
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




if __name__ == '__main__':

    triton_quant_runtimes = []
    had_runtimes = []
    had_with_padding_runtimes = []
    had_with_slicing_runtimes = []
    had_transposed_runtimes = []
    cutlass_runtimes = []
    torch_runtimes = []
    had_group_runtimes = []

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    input_sizes = [(512*8, 4096), (6144, 6144), (8192, 8192), (11008, 11008), (13824, 13824)]#,(4096*4, 4096*4), (28672, 28672), (4096*8, 4096*8)]
    for size in input_sizes:
        in_dim = size[0]
        mid_dim = size[1]
        out_dim = size[1]
        # integer random matrix
        a = torch.randint(2, 10, (in_dim, mid_dim), device='cuda').half()
        b = torch.randint(2, 10, (mid_dim, out_dim), device='cuda').half()
        a_matmul = torch.randint(2, 10, (4096, mid_dim), device='cuda').half()

        triton_quant_time = triton_quant_benchmark(a.detach())
        triton_quant_runtimes.append(np.mean(triton_quant_time))
        
        had_time = had_benchmark(a.detach())
        had_runtimes.append(np.mean(had_time))

        group_had_time = group_had_benchmark(a.detach())
        had_group_runtimes.append(np.mean(group_had_time))

        had_with_padding_time = had_with_padding_benchmark(a.detach())
        had_with_padding_runtimes.append(np.mean(had_with_padding_time))

        had_with_slicing_time = had_with_slicing_benchmark(a.detach())
        had_with_slicing_runtimes.append(np.mean(had_with_slicing_time))

        had_transpose_time = had_transpose_benchmark(a.detach())
        had_transposed_runtimes.append(np.mean(had_transpose_time))

        cutlass_time = cutlass_benchmark(a_matmul.to(torch.int8).detach(), b.to(torch.int8).detach())
        cutlass_runtimes.append(np.mean(cutlass_time))

        torch_time = torch_matmul_benchmark(a_matmul.detach(), b.detach())
        torch_runtimes.append(np.mean(torch_time))


    ax.plot(triton_quant_runtimes, '-o', label='Quantization (Triton)')
    ax.plot(had_runtimes, '-o', label='Hadamard')
    ax.plot(had_with_padding_runtimes, '-o', label='Hadamard with Padding')
    ax.plot(had_with_slicing_runtimes, '-o', label='Hadamard with Slicing')
    ax.plot(had_group_runtimes, '-o', label='Hadamard Group')
    ax.plot(had_transposed_runtimes, '-o', label='Hadamard Transpose')
    ax.plot(cutlass_runtimes, '-o', label='INT8 MM (in_dim: 4096)')
    ax.plot(torch_runtimes, '-o', label='FP16 MM (in_dim: 4096)')
    x_labels = [f'{B}' for B, M in input_sizes]
    ax.set_xticks(range(len(input_sizes)))
    ax.set_xticklabels(x_labels)

    xtick_labels = ax.get_xticklabels()


    # Change the color of specific ticks
    for label in xtick_labels:
        if is_pow2(int(label.get_text())):
            label.set_color('red')


        
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title(f'Kernel Runtimes on {torch.cuda.get_device_name()}')
    ax.legend()


    plt.savefig(f'{torch.cuda.get_device_name()}_kernel_runtimes.png')
