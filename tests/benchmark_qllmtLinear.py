import torch, time, gc
import transformers
import numpy as np
import argparse
import copy
import qllmt

import matplotlib.pyplot as plt
import seaborn as sns

num_warmup_steps = 3
num_bench_steps = 9



def repeated_run(num_repeats=10):
    def func(module):
        def _f(*args, **kwargs):
            times = []
            for i in range(num_repeats):
                times.append(module(*args, **kwargs))
            return times
        return _f
    return func

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


@repeated_run()
def linear_benchmark(module, x):
    # warmup
    for i in range(num_warmup_steps):
        out = module(x)
        out.sum().backward()
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = module(x)
        out.sum().backward()     
    torch.cuda.synchronize()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps

@repeated_run()
def linear_benchmark_fwd(module, x):
    # warmup
    for i in range(num_warmup_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = module(x)
    torch.cuda.synchronize()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps




if __name__ == '__main__':
    
    bits = 16
    batch_sizes = [2048, 4096, 8192, 16384]
    sizes = [(4096, 4096), (4096, 11008), (11008, 4096), (11008, 11008)]
    
    

    # plot settings for batch_sizes*3
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(len(batch_sizes), 3, figsize=(20, 15))
    # super title
    fig.suptitle(f'{torch.cuda.get_device_name()} Benchmark {bits}-bit', fontsize=16)


    for i, bsz in enumerate(batch_sizes):
        torch_runtimes_all = []
        torch_runtimes_fwd = []
        torch_runtimes_bwd = []

        qllmt_linear_runtimes_all = []
        qllmt_linear_runtimes_fwd = []
        qllmt_linear_runtimes_bwd = []

        qllmt_linear_HadY_runtimes_all = []
        qllmt_linear_HadY_runtimes_fwd = []
        qllmt_linear_HadY_runtimes_bwd = []

        qllmt_linear_ExHad_runtimes_all = []
        qllmt_linear_ExHad_runtimes_fwd = []
        qllmt_linear_ExHad_runtimes_bwd = []

        qllmt_linear_YExHad_runtimes_all = []
        qllmt_linear_YExHad_runtimes_fwd = []
        qllmt_linear_YExHad_runtimes_bwd = []
        for j, size in enumerate(sizes):
            torch_linear = torch.nn.Linear(size[0], size[1], bias=False).cuda().half()
            qllmt_linear = qllmt.nn.LinearHad.from_unquantized(torch_linear,
                                                            device=torch_linear.weight.device,
                                                            fwd_bits=bits, bwd_1_bits=bits, bwd_2_bits=bits,
                                                            output_had=False, input_grad_had=False)
            qllmt_linear_HadY = qllmt.nn.LinearHad.from_unquantized(torch_linear,
                                                                device=torch_linear.weight.device,
                                                                fwd_bits=bits, bwd_1_bits=bits, bwd_2_bits=bits,
                                                                output_had=True, input_grad_had=False)
            qllmt_linear_ExHad = qllmt.nn.LinearHad.from_unquantized(torch_linear,
                                                                    device=torch_linear.weight.device,
                                                                    fwd_bits=bits, bwd_1_bits=bits, bwd_2_bits=bits,
                                                                    output_had=False, input_grad_had=True)
            qllmt_linear_YExHad = qllmt.nn.LinearHad.from_unquantized(torch_linear,
                                                                    device=torch_linear.weight.device,
                                                                    fwd_bits=bits, bwd_1_bits=bits, bwd_2_bits=bits,
                                                                    output_had=True, input_grad_had=True)

            # torch experiments
            torch_runtime_all = linear_benchmark(torch_linear, torch.randn(bsz, size[0]).cuda().half())
            torch_runtime_fwd = linear_benchmark_fwd(torch_linear, torch.randn(bsz, size[0]).cuda().half())
            torch_runtime_bwd = np.array(torch_runtime_all) - np.array(torch_runtime_fwd)
            torch_runtimes_all.append(np.mean(torch_runtime_all))
            torch_runtimes_fwd.append(np.mean(torch_runtime_fwd))
            torch_runtimes_bwd.append(np.mean(torch_runtime_bwd))


            # qllmt_linear experiments
            qllmt_linear_runtime_all = linear_benchmark(qllmt_linear, torch.randn(bsz, size[0]).cuda().half())
            qllmt_linear_runtime_fwd = linear_benchmark_fwd(qllmt_linear, torch.randn(bsz, size[0]).cuda().half())
            qllmt_linear_runtime_bwd = np.array(qllmt_linear_runtime_all) - np.array(qllmt_linear_runtime_fwd)
            qllmt_linear_runtimes_all.append(np.mean(qllmt_linear_runtime_all))
            qllmt_linear_runtimes_fwd.append(np.mean(qllmt_linear_runtime_fwd))
            qllmt_linear_runtimes_bwd.append(np.mean(qllmt_linear_runtime_bwd))


            # qllmt_linear_HadY experiments
            qllmt_linear_HadY_runtime_all = linear_benchmark(qllmt_linear_HadY, torch.randn(bsz, size[0]).cuda().half())
            qllmt_linear_HadY_runtime_fwd = linear_benchmark_fwd(qllmt_linear_HadY, torch.randn(bsz, size[0]).cuda().half())
            qllmt_linear_HadY_runtime_bwd = np.array(qllmt_linear_HadY_runtime_all) - np.array(qllmt_linear_HadY_runtime_fwd)
            qllmt_linear_HadY_runtimes_all.append(np.mean(qllmt_linear_HadY_runtime_all))
            qllmt_linear_HadY_runtimes_fwd.append(np.mean(qllmt_linear_HadY_runtime_fwd))
            qllmt_linear_HadY_runtimes_bwd.append(np.mean(qllmt_linear_HadY_runtime_bwd))

            # qllmt_linear_ExHad experiments
            qllmt_linear_ExHad_runtime_all = linear_benchmark(qllmt_linear_ExHad, torch.randn(bsz, size[0]).cuda().half())
            qllmt_linear_ExHad_runtime_fwd = linear_benchmark_fwd(qllmt_linear_ExHad, torch.randn(bsz, size[0]).cuda().half())
            qllmt_linear_ExHad_runtime_bwd = np.array(qllmt_linear_ExHad_runtime_all) - np.array(qllmt_linear_ExHad_runtime_fwd)
            qllmt_linear_ExHad_runtimes_all.append(np.mean(qllmt_linear_ExHad_runtime_all))
            qllmt_linear_ExHad_runtimes_fwd.append(np.mean(qllmt_linear_ExHad_runtime_fwd))
            qllmt_linear_ExHad_runtimes_bwd.append(np.mean(qllmt_linear_ExHad_runtime_bwd))

            # qllmt_linear_YExHad experiments
            qllmt_linear_YExHad_runtime_all = linear_benchmark(qllmt_linear_YExHad, torch.randn(bsz, size[0]).cuda().half())
            qllmt_linear_YExHad_runtime_fwd = linear_benchmark_fwd(qllmt_linear_YExHad, torch.randn(bsz, size[0]).cuda().half())
            qllmt_linear_YExHad_runtime_bwd = np.array(qllmt_linear_YExHad_runtime_all) - np.array(qllmt_linear_YExHad_runtime_fwd)
            qllmt_linear_YExHad_runtimes_all.append(np.mean(qllmt_linear_YExHad_runtime_all))
            qllmt_linear_YExHad_runtimes_fwd.append(np.mean(qllmt_linear_YExHad_runtime_fwd))
            qllmt_linear_YExHad_runtimes_bwd.append(np.mean(qllmt_linear_YExHad_runtime_bwd))


        qllmt_linear_runtimes_fwd = np.array(torch_runtimes_fwd) / np.array(qllmt_linear_runtimes_fwd)
        qllmt_linear_HadY_runtimes_fwd = np.array(torch_runtimes_fwd) / np.array(qllmt_linear_HadY_runtimes_fwd)
        qllmt_linear_ExHad_runtimes_fwd = np.array(torch_runtimes_fwd) / np.array(qllmt_linear_ExHad_runtimes_fwd)
        qllmt_linear_YExHad_runtimes_fwd = np.array(torch_runtimes_fwd) / np.array(qllmt_linear_YExHad_runtimes_fwd)

        qllmt_linear_runtimes_bwd = np.array(torch_runtimes_bwd) / np.array(qllmt_linear_runtimes_bwd)
        qllmt_linear_HadY_runtimes_bwd = np.array(torch_runtimes_bwd) / np.array(qllmt_linear_HadY_runtimes_bwd)
        qllmt_linear_ExHad_runtimes_bwd = np.array(torch_runtimes_bwd) / np.array(qllmt_linear_ExHad_runtimes_bwd)
        qllmt_linear_YExHad_runtimes_bwd = np.array(torch_runtimes_bwd) / np.array(qllmt_linear_YExHad_runtimes_bwd)

        qllmt_linear_runtimes_all = np.array(torch_runtimes_all) / np.array(qllmt_linear_runtimes_all)
        qllmt_linear_HadY_runtimes_all = np.array(torch_runtimes_all) / np.array(qllmt_linear_HadY_runtimes_all)
        qllmt_linear_ExHad_runtimes_all = np.array(torch_runtimes_all) / np.array(qllmt_linear_ExHad_runtimes_all)
        qllmt_linear_YExHad_runtimes_all = np.array(torch_runtimes_all) / np.array(qllmt_linear_YExHad_runtimes_all)
        
        # ax[i][0].plot(torch_runtimes_fwd, '-o', c='C0', label='torch')
        # ax[i][1].plot(torch_runtimes_bwd, '-o', c='C0', label='torch')
        # ax[i][2].plot(torch_runtimes_all, '-o', c='C0',label='torch')

        ax[i][0].plot(qllmt_linear_runtimes_fwd, '-o', c='C1', label='QllmTLinear')
        ax[i][1].plot(qllmt_linear_runtimes_bwd, '-o', c='C1', label='QllmTLinear')
        ax[i][2].plot(qllmt_linear_runtimes_all, '-o', c='C1', label='QllmTLinear')

        ax[i][0].plot(qllmt_linear_HadY_runtimes_fwd, '-o', c='C2', label='QllmTLinear_HadY')
        ax[i][1].plot(qllmt_linear_HadY_runtimes_bwd, '-o', c='C2', label='QllmTLinear_HadY')
        ax[i][2].plot(qllmt_linear_HadY_runtimes_all, '-o', c='C2', label='QllmTLinear_HadY')

        ax[i][0].plot(qllmt_linear_ExHad_runtimes_fwd, '-o', c='C3', label='QllmTLinear_ExHad')
        ax[i][1].plot(qllmt_linear_ExHad_runtimes_bwd, '-o', c='C3', label='QllmTLinear_ExHad')
        ax[i][2].plot(qllmt_linear_ExHad_runtimes_all, '-o', c='C3', label='QllmTLinear_ExHad')

        ax[i][0].plot(qllmt_linear_YExHad_runtimes_fwd, '-o', c='C4', label='QllmTLinear_YExHad')
        ax[i][1].plot(qllmt_linear_YExHad_runtimes_bwd, '-o', c='C4', label='QllmTLinear_YExHad')
        ax[i][2].plot(qllmt_linear_YExHad_runtimes_all, '-o', c='C4', label='QllmTLinear_YExHad')

    
    ax[0][0].legend()
    x_labels = [f'{B}x{M}' for B, M in sizes]
    for i in range(len(batch_sizes)):
        ax[i][0].set_ylabel('speedup (over torch)')
        for j in range(3):
            ax[i][j].set_xticks(range(len(sizes)))
            ax[i][j].set_xticklabels(x_labels)

            ax[i][0].set_title(f'Forward (Batch Dim: {batch_sizes[i]})')
            ax[i][1].set_title(f'Backward (Batch Dim: {batch_sizes[i]})')
            ax[i][2].set_title(f'Forward+Backward (Batch Dim: {batch_sizes[i]})')
   
    plt.savefig(f'{torch.cuda.get_device_name()}_benchmark_qllmtLinear_{bits}bits.png')
            

