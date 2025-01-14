import torch, time, gc
import qllmt
import numpy as np
import argparse
import pprint
from copy import deepcopy
from qllmt.nn.wrapping_utils import wrap_linear_module
from qllmt.nn.jetfire_utils import int8_quantize, int8_dequantize
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json

num_warmup_steps = 100
num_bench_steps = 100


def repeated_run(num_repeats=20):
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
def module_benchmark(module, x, dy):
    if not isinstance(x, tuple):
        x = (x,)
        
    # warmup
    for i in range(num_warmup_steps):
        out = module(*x)
        torch.autograd.backward(out, dy)
        for xi in x:
            xi.grad = None
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = module(*x)
        torch.autograd.backward(out, dy)
        for xi in x:
            xi.grad = None
    torch.cuda.synchronize()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps

def error(x, y):
    return torch.norm(x - y).item() / torch.norm(y).item()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Argument parser for model parameters and benchmark settings.")

    # Add arguments with default values
    parser.add_argument('--in_dim', type=int, default=4096, help='Input dimension size.')
    parser.add_argument('--out_dim', type=int, default=4096, help='Output dimension size.')
    
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length.')

    parser.add_argument('--runtime_benchmark', type=bool, default=True, help='Enable runtime benchmarking.')
    parser.add_argument('--accuracy_benchmark', type=bool, default=True, help='Enable accuracy benchmarking.')

    parser.add_argument('--kernels', nargs='+', help='kernels to compare.', default=['base'])

    args = parser.parse_args()
    pprint.pprint(args.__dict__)
    print()
    
    bsz_list = [4, 8, 16, 32]

    seq_len = args.seq_len
    in_dim = args.in_dim
    out_dim = args.out_dim
    runtime_benchmark = args.runtime_benchmark
    accuracy_benchmark = args.accuracy_benchmark
    
    dtype = torch.bfloat16
    # dtype = torch.float16
    device = 'cuda:0'
    
    base_module_orig = torch.nn.Linear(in_dim, out_dim, dtype=dtype, device=device, bias=False)

    modules = []
    kernels = args.kernels
    for kernel in kernels:
        if kernel == 'base':
            modules.append(deepcopy(base_module_orig))
        else:
            hq_config = {'kernel': kernel}
            modules.append(wrap_linear_module(deepcopy(base_module_orig), hq_config))

    speedups_all = [[] for _ in modules]
    stds_all = [[] for _ in modules]
    if runtime_benchmark:
        for bsz in bsz_list:
            x_orig = torch.randn(bsz, seq_len, in_dim, dtype=dtype, device=device, requires_grad=True)
            dy = torch.randn(bsz, seq_len, out_dim, dtype=dtype, device=device, requires_grad=False)
            qx_orig, sx_orig = int8_quantize(x_orig.detach().clone(), 32)
            qx_orig = qx_orig.view(torch.float8_e4m3fn)
            qdy, sdy = int8_quantize(dy, 32)
            qdy = qdy.view(torch.float8_e4m3fn)

            times_all = []
            for kernel, module in zip(kernels, modules):
                if 'jetfire_real' in kernel:
                    qx = qx_orig.detach().clone()
                    sx = sx_orig.detach().clone()
                    qx.requires_grad_()
                    sx.requires_grad_()
                    time_all = module_benchmark(module, (qx, sx), (qdy, sdy))
                else:
                    x = x_orig.detach().clone()
                    x.requires_grad_()
                    time_all = module_benchmark(module, x, dy)
                times_all.append(time_all)

            for i in range(len(modules)):
                speedups_all[i].append(np.mean(times_all[0]) / np.mean(times_all[i]))
                stds_all[i].append(np.std(np.mean(times_all[0]) / np.array(times_all[i])))
        
            for name, tall in zip(kernels, times_all):
                print(f"{name} time (all): {np.mean(tall):.3f} +- {1.96 * np.std(tall):.3f}ms    --> {np.mean(times_all[0]) / np.mean(tall):.2f}x")
    
    print('\n' + '-' * 80)

    bsz = 16
    x_orig = torch.randn(bsz, seq_len, in_dim, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(bsz, seq_len, out_dim, dtype=dtype, device=device, requires_grad=False)
    qx_orig, sx_orig = int8_quantize(x_orig.detach().clone(), 32)
    qx_orig = qx_orig.view(torch.float8_e4m3fn)
    qdy, sdy = int8_quantize(dy, 32)
    qdy = qdy.view(torch.float8_e4m3fn)

    if accuracy_benchmark:
        outs = []
        grad_xs = []
        grad_ws = []
        for kernel, module in zip(kernels, modules):
            module.zero_grad()

            if 'jetfire_real' in kernel:
                qx = qx_orig.detach().clone()
                sx = sx_orig.detach().clone()
                qx.requires_grad_()
                sx.requires_grad_()
                qy, sy = module(qx, sx)
                torch.autograd.backward((qy, sy), (qdy, sdy))
                qy = qy.view(torch.int8)
                y = int8_dequantize(qy, sy, 32)
                outs.append(y.detach())
                grad_xs.append(int8_dequantize(qx.grad.detach().view(torch.int8), sx.grad.detach(), 32))
                grad_ws.append(module.weight.grad.detach())
            else:
                x = x_orig.detach().clone()
                x.requires_grad_()
                y = module(x)
                y.backward(dy)
                outs.append(y.detach())
                grad_xs.append(x.grad.detach())
                grad_ws.append(module.weight.grad.detach())


        for name, out in zip(kernels, outs):
            print(f"{name} error (fwd): {error(out, outs[0])}")
        print()
        for name, grad_x in zip(kernels, grad_xs):
            print(f"{name} error (bwd1): {error(grad_x, grad_xs[0])}")
        print()
        for name, grad_w in zip(kernels, grad_ws):
            print(f"{name} error (bwd2): {error(grad_w, grad_ws[0])}")

    # name_map = {
    #     'base': 'BF16' if dtype == torch.bfloat16 else 'FP16',
    #     'jetfire_real': 'Jetfire',
    #     'switchback': 'SwitchBack',
    #     'halo0_fp8': 'HALO-0 FP8',
    #     'halo1_fp8': 'HALO-1 FP8',
    #     'halo2_fp8': 'HALO-2 FP8',
    #     'haloi_fp8': 'Ideal FP8',
    #     'halo0_fp8_qfsdp': 'HALO-0 FP8 HQ-FSDP',
    #     'halo1_fp8_qfsdp': 'HALO-1 FP8 HQ-FSDP',
    #     'halo2_fp8_qfsdp': 'HALO-2 FP8 HQ-FSDP',
    #     'haloi_fp8_qfsdp': 'Ideal FP8 HQ-FSDP',
    #     'halo0_int8': 'HALO-0 INT8',
    #     'halo1_int8': 'HALO-1 INT8',
    #     'halo2_int8': 'HALO-2 INT8',
    #     'haloi_int8': 'Ideal INT8',
    #     'halo0_int8_qfsdp': 'HALO-0 INT8 HQ-FSDP',
    #     'halo1_int8_qfsdp': 'HALO-1 INT8 HQ-FSDP',
    #     'halo2_int8_qfsdp': 'HALO-2 INT8 HQ-FSDP',
    #     'haloi_int8_qfsdp': 'Ideal INT8 HQ-FSDP',
    # }

    name_map = {
        'base': 'BF16' if dtype == torch.bfloat16 else 'FP16',
        'jetfire_real': 'Jetfire',
        'switchback': 'SwitchBack',
        'halo0_fp8': 'HALO-0',
        'halo1_fp8': 'HALO-1',
        'halo2_fp8': 'HALO-2',
        'haloi_fp8': 'Ideal',
        'halo0_fp8_qfsdp': 'HALO-0 HQ-FSDP',
        'halo1_fp8_qfsdp': 'HALO-1 HQ-FSDP',
        'halo2_fp8_qfsdp': 'HALO-2 HQ-FSDP',
        'haloi_fp8_qfsdp': 'Ideal HQ-FSDP',
        'halo0_int8': 'HALO-0',
        'halo1_int8': 'HALO-1',
        'halo2_int8': 'HALO-2',
        'haloi_int8': 'Ideal',
        'halo0_int8_qfsdp': 'HALO-0 HQ-FSDP',
        'halo1_int8_qfsdp': 'HALO-1 HQ-FSDP',
        'halo2_int8_qfsdp': 'HALO-2 HQ-FSDP',
        'haloi_int8_qfsdp': 'Ideal HQ-FSDP',
    }

    def theme():
        # Apply the default theme
        # sns.set_theme()
        sns.set_theme(style="whitegrid", palette="colorblind")
        sns.set(font_scale=1.5)
        # sns.set_style("darkgrid", {"axes.facecolor": ".95"})
        # set fonttype
        # matplotlib.rcParams['pdf.fonttype'] = 42
        # matplotlib.rcParams['ps.fonttype']  = 42
        # matplotlib.rcParams['font.family'] = 'serif'
            
        plt.rcParams.update({
            'font.family': 'sans-serif',  # Use sans-serif fonts
            'axes.linewidth': 1.0,  # Border line width
            'lines.linewidth': 2.0,  # Line width
            'lines.markersize': 6,  # Marker size
            'grid.alpha': 0.5,  # Gridline transparency
            'savefig.dpi': 300,  # DPI for saving figures
            'savefig.format': 'pdf'  # Save as vector format
        })

        # matplotlib.rcParams.update({
        #     'font.size': 10,  # Base font size
        #     'font.family': 'sans-serif',  # Use sans-serif fonts
        #     'axes.labelsize': 12,  # Label font size
        #     'axes.titlesize': 14,  # Title font size
        #     'axes.linewidth': 1.0,  # Border line width
        #     'lines.linewidth': 2.0,  # Line width
        #     'lines.markersize': 6,  # Marker size
        #     'xtick.labelsize': 10,  # X-axis tick font size
        #     'ytick.labelsize': 10,  # Y-axis tick font size
        #     'xtick.major.size': 5,  # Major tick size
        #     'ytick.major.size': 5,
        #     'legend.fontsize': 10,  # Legend font size
        #     # 'legend.frameon': False,  # No frame for legend
        #     'grid.alpha': 0.5,  # Gridline transparency
        #     'figure.figsize': (6, 4),  # Default figure size
        #     'savefig.dpi': 300,  # DPI for saving figures
        #     'savefig.format': 'pdf'  # Save as vector format
        # })

    def barplot(names, speedups, title, file_name, figsize=(10, 6), ylabel=None):
        theme()

        if ylabel is None:
            ylabel = f'Speedup Over BF16'

        # throw away the base
        names = names[1:]
        speedups = speedups[1:]

        # Number of implementations and batch sizes
        n_names = len(names)
        n_bsz = len(bsz_list)
        
        # Set up the positions for the bars
        width = 0.8 / n_names  # Width of each bar
        positions = np.arange(n_bsz)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bars for each implementation
        for i, (name, speedup_values) in enumerate(zip(names, speedups)):
            x_positions = positions + (i - n_names/2 + 0.5) * width
            bars = ax.bar(x_positions, speedup_values, width, label=name_map[name])
            
            # # Add value labels on top of bars
            # for bar in bars:
            #     height = bar.get_height()
            #     ax.text(bar.get_x() + bar.get_width()/2, height,
            #         f'{height:.2f}',
            #         ha='center', va='bottom')
        
        # Customize the plot
        ax.set_ylabel(ylabel)
        # ax.set_title(title)
        ax.set_xticks(positions)
        ax.set_xticklabels([f'BS={bsz}' for bsz in bsz_list])
        ax.axhline(1, linestyle='--', color=('black', 0.7), label='BF16')
        ax.set_ylim(ymin=0.5)
        
        # Add legend
        ax.legend(loc='lower right')
        
        # Add grid for better readability
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent label clipping
        plt.tight_layout()
        
        fig.savefig(file_name)
        fig.clf()

    file_name = f"4090_linear_benchmark_{dtype}"
    title = f'Linear Module on 4090 with {dtype}'

    barplot(
        names=kernels,
        speedups=speedups_all,
        title=f'{title} - Fwd+Bwd',
        file_name=f'{file_name}_all.pdf'
    )

    output = {
        'speedups': speedups_all,
        'stds': stds_all,
        'names': [name_map[name] for name in kernels],
        'bsz_list': bsz_list
    }

    with open(f'{file_name}.json', 'w') as f:
        json.dump(output, f, indent=4)
