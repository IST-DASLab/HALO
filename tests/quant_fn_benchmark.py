import qllmt

import argparse, pprint, torch, time, gc
import numpy as np
num_warmup_steps = 1
num_bench_steps = 10



def triton_per_tensor_int8_quant_transpose_triton(x):
    kernel_times = []
    for i in range(num_warmup_steps):
        out, scale = qllmt.functional.per_tensor_int8_quant_transpose_triton(x.clone())
        # print(scale)
        # print(out[:3, :3])
        torch.cuda.synchronize()
    for i in range(10):
        start_time = time.perf_counter()
        for i in range(num_bench_steps):
            out, scale = qllmt.functional.per_tensor_int8_quant_transpose_triton(x.clone())
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        kernel_times.append((end_time - start_time) * 1000 / num_bench_steps)
    return kernel_times


def triton_per_tensor_int8_quant(x):
    kernel_times = []
    for i in range(num_warmup_steps):
        out, scale = qllmt.functional.per_tensor_int8_quant_triton(x)
        # print(scale)
        # print(out[:3, :3])
        torch.cuda.synchronize()
    for i in range(10):
        start_time = time.perf_counter()
        for i in range(num_bench_steps):
            out, scale = qllmt.functional.per_tensor_int8_quant_triton(x)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        kernel_times.append((end_time - start_time) * 1000 / num_bench_steps)
    return kernel_times

def torch_per_tensor_int8_quant(x):
    kernel_times = []
    for i in range(num_warmup_steps):
        out, scale = qllmt.functional.per_tensor_int8_quant(x)
        # print(scale)
        # print(out[:3, :3])
        torch.cuda.synchronize()
    for i in range(10):
        start_time = time.perf_counter()
        for i in range(num_bench_steps):
            out = qllmt.functional.per_tensor_int8_quant(x)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        kernel_times.append((end_time - start_time) * 1000 / num_bench_steps)
    return kernel_times    
     
    
    
def torch_per_tensor_int8_quant_v2(x):
    kernel_times = []
    for i in range(num_warmup_steps):
        out, scale = qllmt.functional.per_tensor_int8_quant_v2(x)
        # print(scale)
        # print(out[:3, :3])
        torch.cuda.synchronize()
    for i in range(10):
        start_time = time.perf_counter()
        for i in range(num_bench_steps):
            out = qllmt.functional.per_tensor_int8_quant_v2(x)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        kernel_times.append((end_time - start_time) * 1000 / num_bench_steps)
    return kernel_times    
    
    
def torch_transpose(x):
    kernel_times = []
    for i in range(num_warmup_steps):
        out = x.mT.contiguous()
        # print(out[:3, :3])
        torch.cuda.synchronize()
    for i in range(10):
        start_time = time.perf_counter()
        for i in range(num_bench_steps):
           out = x.mT.contiguous()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        kernel_times.append((end_time - start_time) * 1000 / num_bench_steps)
    return kernel_times    
    
    
def runtime_with_plot():
    import matplotlib.pyplot as plt
    input_sizes = [(512*8, 4096), (512*8, 8192), (512*8, 11008), (512*8, 28672)]
    torch_quant_times = []
    triton_quant_times = []
    triton_quant_transpose_times = []
    torch_transpose_times = []
    torch_per_tensor_int8_quant_v2_times = []
    for B, M in input_sizes:
        x = torch.rand(B, M, device='cuda').half()
        torch_quant_time = torch_per_tensor_int8_quant(x.detach().clone())
        triton_quant_time = triton_per_tensor_int8_quant(x.detach().clone())
        triton_quant_transpose_time = triton_per_tensor_int8_quant_transpose_triton(x.detach().clone())
        torch_per_tensor_int8_quant_v2_time = torch_per_tensor_int8_quant_v2(x.detach().clone())
        x = x.to(torch.int8)
        torch_transpose_time = torch_transpose(x.detach().clone())
        torch_transpose_times.append(np.mean(torch_transpose_time))
        torch_quant_times.append(np.mean(torch_quant_time))
        triton_quant_times.append(np.mean(triton_quant_time))
        triton_quant_transpose_times.append(np.mean(triton_quant_transpose_time))
        torch_per_tensor_int8_quant_v2_times.append(np.mean(torch_per_tensor_int8_quant_v2_time))
    # plot results
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(torch_quant_times, label='Torch Quant')
    ax.plot(triton_quant_times, label='Triton Quant')
    ax.plot(triton_quant_transpose_times, label='Triton Quant Transpose')
    ax.plot(torch_transpose_times, label='Torch INT8 Transpose')
    ax.plot(torch_per_tensor_int8_quant_v2_times, label='Torch Quant v2')
    
    # put input_sizes in the x_labels
    x_labels = [f'{B}x{M}' for B, M in input_sizes]
    ax.set_xticks(range(len(input_sizes)))
    ax.set_xticklabels(x_labels)
    
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Runtime Benchmark GPU: {torch.cuda.get_device_name()}')
    ax.legend()
    plt.savefig(f'{torch.cuda.get_device_name()}_quantization_runtime.png')
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Argument parser for model parameters and benchmark settings.")

    # Add arguments with default values
    parser.add_argument('--in_dim', type=int, default=4096, help='Input dimension size.')
    parser.add_argument('--out_dim', type=int, default=4096, help='Output dimension size.')
    
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length.')
    parser.add_argument('--bsz', type=int, default=8, help='Batch size.')

    parser.add_argument('--runtime_benchmark', type=bool, default=True, help='Enable runtime benchmarking.')

    args = parser.parse_args()
    
    if args.runtime_benchmark:
        runtime_with_plot()
        exit()
    
    pprint.pprint(args.__dict__)
    
    B = args.bsz * args.seq_len
    M = args.in_dim
    N = args.out_dim
    x = torch.rand(B, M, device='cuda').half()

    torch_quant_time = torch_per_tensor_int8_quant(x.detach().clone())
    print(f"Torch Quant time: {np.mean(torch_quant_time):.3f} +- {1.96 * np.std(torch_quant_time):.3f}ms")
    torch_quant_v2_time = torch_per_tensor_int8_quant_v2(x.detach().clone())
    print(f"Torch Quant v2 time: {np.mean(torch_quant_v2_time):.3f} +- {1.96 * np.std(torch_quant_v2_time):.3f}ms")
    triton_quant_time = triton_per_tensor_int8_quant(x.detach().clone())
    print(f"Triton Quant time: {np.mean(triton_quant_time):.3f} +- {1.96 * np.std(triton_quant_time):.3f}ms")
    triton_quant_transpose_time = triton_per_tensor_int8_quant_transpose_triton(x.detach().clone())
    print(f"Triton Quant Transpose time: {np.mean(triton_quant_transpose_time):.3f} +- {1.96 * np.std(triton_quant_transpose_time):.3f}ms")
    
    x = x.to(torch.int8)
    torch_transpose_time = torch_transpose(x.detach().clone())
    print(f"Torch Transpose time: {np.mean(torch_transpose_time):.3f} +- {1.96 * np.std(torch_transpose_time):.3f}ms")
    
    
