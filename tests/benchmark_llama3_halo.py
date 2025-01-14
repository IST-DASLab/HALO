# run the following for FSDP benchmarking:
# NCCL_NTHREADS=64 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 benchmark_llama_multi.py --fsdp --num_blocks 9

# and for single-node benchmarking:
# CUDA_VISIBLE_DEVICES=0 python benchmark_llama_multi.py --num_blocks 9 --hq_schemes base mlsys-int8-global

import torch, time, gc
import transformers
import numpy as np
import argparse
import os
import copy
import json

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from qllmt.nn.wrapping_utils import wrap_model
import qllmt

import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

def dist_setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def dist_cleanup():
    dist.destroy_process_group()


num_warmup_steps = 3
num_bench_steps = 10


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass


def repeated_run(num_repeats=3):
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
def llama_decoder_benchmark(module, x, position_ids, fsdp=False):
    if not fsdp:
        module.to(device='cuda')
    x = x.cuda(dist.get_rank() if fsdp else 0)

    # warmup
    for i in range(num_warmup_steps):
        out = module(x, position_ids=position_ids)[0]
        out.sum().backward()
        if fsdp:
            dist.barrier()
    torch.cuda.synchronize()
    if fsdp:
        dist.barrier()

    start_time = time.perf_counter()

    for i in range(num_bench_steps):
        out = module(x, position_ids=position_ids)[0]
        out.sum().backward()
        if fsdp:
            dist.barrier()
    torch.cuda.synchronize()
    if fsdp:
        dist.barrier()

    end_time = time.perf_counter()
    
    if not fsdp:
        module.to(device='cpu')

    return (end_time - start_time) * 1000 / num_bench_steps


@repeated_run()
def llama_decoder_benchmark_fwd(module, x, position_ids, fsdp=False):
    if not fsdp:
        module.to(device='cuda')
    x = x.cuda(dist.get_rank() if fsdp else 0)

    halo_modules = [m for m in module.modules() if hasattr(m, 'hq_config')]
    with torch.no_grad():
        # warmup
        for i in range(num_warmup_steps):
            out = module(x, position_ids=position_ids)[0]
            if fsdp:
                dist.barrier()
                for m in halo_modules:
                    qllmt.nn.fsdp.qfsdp_backward(m.hq_config)
        torch.cuda.synchronize()
        if fsdp:
            dist.barrier()

        start_time = time.perf_counter()

        for i in range(num_bench_steps):
            out = module(x, position_ids=position_ids)[0]
            if fsdp:
                dist.barrier()
                for m in halo_modules:
                    qllmt.nn.fsdp.qfsdp_backward(m.hq_config)
        torch.cuda.synchronize()
        if fsdp:
            dist.barrier()

        end_time = time.perf_counter()

    if not fsdp:
        module.to(device='cpu')

    return (end_time - start_time) * 1000 / num_bench_steps


def get_llama3_8b(torch_dtype):
    model = transformers.LlamaForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct',
                                                          use_cache=False,
                                                          torch_dtype=torch_dtype,
                                                          attn_implementation="flash_attention_2", )
    return model


def prepare_num_blocks(model, num_blocks, keep_lm_head=False):
    model.model.layers = model.model.layers[:num_blocks]

    # replace embedding with identity
    model.model.embed_tokens = torch.nn.Identity()

    if not keep_lm_head:
        model.model.norm = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()

    return model

def swap_module(network, module_name, new_module):
    name_parts = module_name.split('.')
    parent = network
    for part in name_parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last_part = name_parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)


def run_fwd_bwd(module, x, position_ids, fsdp=False):
    if not fsdp:
        module.to(device='cuda')
        
    x_cpy = x.clone()
    x_cpy = x_cpy.cuda(dist.get_rank() if fsdp else 0)
    x_cpy.requires_grad_()
    x_cpy.retain_grad()
    setattr(x_cpy, 'grad', None)
    module.zero_grad()
    out = module(x_cpy, position_ids=position_ids)[0]
    out.sum().backward()
    
    res = out.detach().cpu(), torch.cat(
        [param.grad.clone().view(-1) for param in module.parameters()]).detach().cpu(), x_cpy.grad.detach().cpu()
    
    if not fsdp:
        module.to(device='cpu')
    return res


def error(x, y):
    return torch.norm(x - y).item() / torch.norm(y).item()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Argument parser for model parameters and benchmark settings.")

    parser.add_argument('--seq_len',
                        type=int,
                        default=512,
                        help='Sequence length.')

    # parser.add_argument('--num_bench_steps',
    #                     type=int,
    #                     default=5,
    #                     help='Number of benchmark steps.')
    
    # parser.add_argument('--num_warmup_steps',
    #                     type=int,
    #                     default=2,
    #                     help='Number of warmup steps.')

    parser.add_argument('--fsdp',
                        default=False,
                        action='store_true',
                        help='Whether to use FSDP.')

    parser.add_argument('--keep_lm_head',
                        default=False,
                        action='store_true',
                        help='Whether to keep the lm_head layer.')

    parser.add_argument('--num_blocks',
                        type=int,
                        default=1,
                        help='Number of transformer blocks to benchmark.')

    parser.add_argument('--kernels', nargs='+', help='kernels to compare.', default=['base'])

    parser.add_argument('--tag',
                        type=str,
                        default=None,
                        help='A tag to include in the plot name.')

    # parser.add_argument('--include_non_patched_qfsdp_halos',
    #                     default=False,
    #                     action='store_true',
    #                     help='Whether to include non-patched versions of qfsdp halo modules.')

    args = parser.parse_args()
    assert args.kernels[0] == 'base', 'The first kernel must be the base model.'
    # num_bench_steps = args.num_bench_steps
    # num_warmup_steps = args.num_warmup_steps
    # print(f'running with {num_warmup_steps} warmup + {num_bench_steps} benchmark steps')

    bits = 8
    # bsz_list = [8]
    bsz_list = [4, 8, 16, 32]
    dtype = 'bf16'

    assert dtype in ['bf16']
    torch_dtype = torch.bfloat16

    seq_len = args.seq_len

    baseline_model_org = prepare_num_blocks(
        get_llama3_8b(torch_dtype),
        num_blocks=args.num_blocks,
        keep_lm_head=args.keep_lm_head
    )

    modules = []
    names = []
    module_args = []

    
    for kernel in args.kernels:
        assert kernel in ['base', 'jetfire_real', 'switchback'] or 'halo' in kernel, f'Unsupported kernel: {kernel}'
        if kernel == 'base':
            model = copy.deepcopy(baseline_model_org)
        else:
            hq_config = {'kernel': kernel}
            model = wrap_model(copy.deepcopy(baseline_model_org), hq_config)

        modules.append(model)
        names.append(kernel)
        # print(model)

        if 'qfsdp' in kernel:
            assert args.fsdp
            assert 'halo' in kernel, 'Only Halo kernels are supported for qfsdp.'
            module_args.append({'qfsdp': True})

            # if args.include_non_patched_qfsdp_halos:
            #     modules.append(wrap_model(copy.deepcopy(baseline_model_org), hq_config))
            #     names.append(kernel + '_non_patched')
            #     module_args.append({'qfsdp': False})
        else:
            module_args.append({'qfsdp': False})

    world_size = 1
    if args.fsdp:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist_setup(rank, world_size)

        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
        torch.cuda.set_device(rank)

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        for i, module in enumerate(modules):
            qfsdp = module_args[i]['qfsdp']
            
            ignored_modules = [m for n, m in modules[i].named_modules() if 'norm' in n] if qfsdp else None
            if ignored_modules is not None:
                for m in ignored_modules:
                    m.to(device=torch.cuda.current_device())
            # print('ignored_modules:', ignored_modules)

            try:
                block_cls = model.model.layers[0].__class__
                # print(f'found block class: {block_cls}')
            except:
                raise ValueError('Unable to find the class for transformer blocks.')
            def auto_wrap_policy(module, recurse, nonwrapped_numel):
                return isinstance(module, block_cls)
        
            modules[i] = FSDP(
                module,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mp_policy,
                forward_prefetch=True,
                device_id=torch.cuda.current_device(),
                ignored_modules=ignored_modules
            )
            
            if qfsdp:
                apply_had = 'halo1' in names[i] or 'halo2' in names[i]
                assert 'fp8' in names[i] or 'int8' in names[i], f'Unsupported precision: {names[i]}'
                halo_precision = 'fp8' if 'fp8' in names[i] else 'int8'
                halo_dtype = qllmt.nn.halo_helpers._precision_to_dtype(halo_precision)
                qllmt.nn.patch_fsdp_model(modules[i], qdtype=halo_dtype, apply_had=apply_had)
            # print(modules[i])
    # else:
    #     for module in modules:
    #         module.cuda()

    # make sure everything is on CPU if FSDP is not enabled to save memory
    if not args.fsdp:
        for module in modules:
            module.to(device='cpu')

    rank = dist.get_rank() if args.fsdp else 0
    world_size = dist.get_world_size() if args.fsdp else 1
    device_name = torch.cuda.get_device_name().split()[-1]

    speedups_all = [[] for _ in modules]
    speedups_fwd = [[] for _ in modules]
    speedups_bwd = [[] for _ in modules]
    stds_all = [[] for _ in modules]

    for bsz in bsz_list:
        x_org = torch.randn(bsz, seq_len, baseline_model_org.config.hidden_size, requires_grad=True,
                            dtype=torch_dtype)
        xs = [torch.empty_like(x_org).copy_(x_org).to(torch_dtype) for _ in modules]
        for x in xs:
            x.requires_grad_()

        position_ids = torch.arange(seq_len).unsqueeze(0).cuda()

        times_all = [llama_decoder_benchmark(module, x, position_ids, fsdp=args.fsdp) for x, module in zip(xs, modules)]
        times_fwd = [llama_decoder_benchmark_fwd(module, x, position_ids, fsdp=args.fsdp) for x, module in
                     zip(xs, modules)]

        outs_grads = [run_fwd_bwd(module, x, position_ids=position_ids, fsdp=args.fsdp) for x, module in zip(xs, modules)]
        outs = [out for out, _, _ in outs_grads]
        grad_ws = [grad_w for _, grad_w, _ in outs_grads]
        grad_xs = [grad_x for _, _, grad_x in outs_grads]

        if rank == 0:
            times_bwd = [np.array(tall) - np.array(tfwd) for tall, tfwd in zip(times_all, times_fwd)]

            print('-----------------')
            print(
                f"[batch size: {bsz}, {world_size} x {device_name} GPUs, {args.num_blocks} transformer blocks, seq len: {seq_len}]")

            for name, tfwd in zip(names, times_fwd):
                print(
                    f"{name} time (fwd): {np.mean(tfwd):.3f} +- {1.96 * np.std(tfwd):.3f}ms    --> {np.mean(times_fwd[0]) / np.mean(tfwd):.2f}x")
            print()
            for name, tbwd in zip(names, times_bwd):
                print(
                    f"{name} time (bwd): {np.mean(tbwd):.3f} +- {1.96 * np.std(tbwd):.3f}ms    --> {np.mean(times_bwd[0]) / np.mean(tbwd):.2f}x")
            print()
            for name, tall in zip(names, times_all):
                print(
                    f"{name} time (all): {np.mean(tall):.3f} +- {1.96 * np.std(tall):.3f}ms    --> {np.mean(times_all[0]) / np.mean(tall):.2f}x")
            print()
            for name, out in zip(names, outs):
                print(f"{name} error (fwd): {error(out, outs[0])}")
            print()
            for name, grad_x in zip(names, grad_xs):
                print(f"{name} error (bwd1): {error(grad_x, grad_xs[0])}")

            # for name, grad_w in zip(names, grad_ws):
            #     print(f"{name} error (bwd2): {error(grad_w, grad_ws[0])}")

            for i in range(len(modules)):
                speedups_all[i].append(np.mean(times_all[0]) / np.mean(times_all[i]))
                speedups_fwd[i].append(np.mean(times_fwd[0]) / np.mean(times_fwd[i]))
                speedups_bwd[i].append(np.mean(times_bwd[0]) / np.mean(times_bwd[i]))
                stds_all[i].append(np.std(np.mean(times_all[0]) / np.array(times_all[i])))


        # del baseline_model, had_model
        del x, xs, position_ids
        gc.collect()

    if args.fsdp:
        dist.barrier()
        dist_cleanup()

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
        'base': 'BF16',
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

    if rank == 0:
        def theme():
            # Apply the default theme
            # sns.set_theme()
            sns.set_theme(style="whitegrid", palette="colorblind")
            sns.set(font_scale=1.5)
            sns.set_style("darkgrid", {"axes.facecolor": ".95"})
            # set fonttype
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype']  = 42
            matplotlib.rcParams['font.family'] = 'serif'

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

        # def barplot(names, speedups, title, file_name):
        #     theme()
        #     for name, spfwd in zip(names[1:], speedups[1:]):
        #         plt.plot(bsz_list, spfwd, '-o', label=name_map[name.lower()])

        #     plt.axhline(1, color='black', linestyle='--')
        #     plt.xlabel('Batch size')
        #     plt.ylabel(f'Speedup (over {dtype.upper()})')
        #     plt.legend()
        #     plt.title(title)
        #     plt.savefig(file_name)
        #     plt.close()

        def barplot(names, speedups, title, file_name, figsize=(10, 6), ylabel='Speedup Over BF16'):
            """
            Create a grouped bar plot comparing speedups across different implementations and batch sizes.
            
            Parameters:
            -----------
            names : list
                Names of the different implementations
            speedups : list of lists
                Each inner list contains speedups for different batch sizes for one implementation
            bsz_list : list
                List of batch sizes
            figsize : tuple, optional
                Figure size (width, height)
            ylabel : str, optional
                Label for y-axis
            title : str, optional
                Plot title
            """
            theme()

            # skip base
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
                bars = ax.bar(x_positions, speedup_values, width, label=name_map[name.lower()])
                
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
            ax.set_ylim(ymin=0.7)

            # Add legend
            ax.legend(loc='lower right')
            
            # Add grid for better readability
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Adjust layout to prevent label clipping
            plt.tight_layout()
            
            fig.savefig(file_name)
            fig.clf()

        fsdp_str = 'w/ FSDP' if args.fsdp else 'w/o FSDP'
        tag_str = '' if args.tag is None else (' ' + args.tag)
        file_name = f"{device_name}_llama3_{args.num_blocks}blocks_{fsdp_str.replace('/', '')}{tag_str}".replace(' ', '_')
        title = f'{args.num_blocks} Blocks on {world_size} {device_name}s {fsdp_str}{tag_str}'

        barplot(
            names=names,
            speedups=speedups_fwd,
            title=f'{title} - Fwd',
            file_name=f'{file_name}_fwd.pdf'
        )

        barplot(
            names=names,
            speedups=speedups_bwd,
            title=f'{title} - Bwd',
            file_name=f'{file_name}_bwd.pdf'
        )

        barplot(
            names=names,
            speedups=speedups_all,
            title=f'{title} - Fwd+Bwd',
            file_name=f'{file_name}_all.pdf'
        )

        output = {
            'speedups': speedups_all,
            'stds': stds_all,
            'names': [name_map[name] for name in names],
            'bsz_list': bsz_list
        }

        with open(f'{file_name}.json', 'w') as f:
            json.dump(output, f, indent=4)