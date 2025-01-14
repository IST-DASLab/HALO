import qllmt
import torch
num_warmup_steps = 3
num_bench_steps = 10
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
@torch.no_grad()
def benchmark_fused_hadq(a):
    # warmup
    for i in range(num_warmup_steps):
        out = qllmt.power_two_had_int8(a)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = qllmt.power_two_had_int8(a)
    torch.cuda.synchronize()

    end_time = time.perf_counter()
    return (end_time - start_time) * 1000 / num_bench_steps


@torch.no_grad()
def benchmark_hadq(a):
    # warmup
    for i in range(num_warmup_steps):
        out = qllmt.functional.per_tensor_int8_quant_v2(qllmt.power_two_had(a))
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = qllmt.functional.per_tensor_int8_quant_v2(qllmt.power_two_had(a))
    torch.cuda.synchronize()

    end_time = time.perf_counter()
    return (end_time - start_time) * 1000 / num_bench_steps




if __name__ == '__main__':
    batch_sizes = [512, 2048, 8192, 32768]
    batch_size_to_color_map = {8: 'r', 32: 'g', 128: 'b', 512: 'c', 2048: 'm', 8192: 'y', 32768: 'k'}
    inner_dim = [512, 1024, 2048, 4096, 8192, 16384, 16384*2]
    times = [[] for i in range(len(batch_sizes))]
    times_fused = [[] for i in range(len(batch_sizes))]
    for idx, bsz in enumerate(batch_sizes):
        for dim in inner_dim:
            a = torch.randn(bsz, dim, device='cuda').half()
            times[idx].append(benchmark_hadq(a))
            times_fused[idx].append(benchmark_fused_hadq(a))

    for idx in range(len(times)):
        ratio = np.array(times[idx])/np.array(times_fused[idx]) 
        plt.plot(inner_dim, ratio, 'o--',label=f'BSZ:{batch_sizes[idx]}')
    # plot y=1
    plt.plot(inner_dim, np.ones_like(inner_dim), 'k--')
    plt.xscale('log')
    plt.xlabel('Hadamard Dim')
    # put xlabels to custom array
    plt.xticks(inner_dim, inner_dim)
    plt.title(f'GPU Type: {torch.cuda.get_device_name()}')

    plt.ylabel('Speedup (over Hadamard+Quant)')
    plt.legend()
    plt.savefig(f'{torch.cuda.get_device_name()}_fused_hadamard.png')