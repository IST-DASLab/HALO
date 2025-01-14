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

from qllmt.nn.wrapping_utils import wrap_model
import qllmt

from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaRMSNorm, LlamaSdpaAttention, \
    LlamaMLP

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


num_warmup_steps = 2
num_bench_steps = 5


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
    # warmup
    for i in range(num_warmup_steps):
        out = module(x, position_ids=position_ids)[0]
        out.sum().backward()
        if fsdp:
            dist.barrier()
    torch.cuda.synchronize()
    if fsdp:
        dist.barrier()

    # module_profilers = attach_model_profilers(module, ".*")
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
    # detach_model_profilers(module_profilers)

    return (end_time - start_time) * 1000 / num_bench_steps


@repeated_run()
def llama_decoder_benchmark_fwd(module, x, position_ids, fsdp=False):
    with torch.no_grad():
        # warmup
        for i in range(num_warmup_steps):
            out = module(x, position_ids=position_ids)[0]
            if fsdp:
                dist.barrier()
        torch.cuda.synchronize()
        if fsdp:
            dist.barrier()

        start_time = time.perf_counter()

        for i in range(num_bench_steps):
            # if i == 0:
            #     with torch.profiler.profile() as p:
            #         out = module(x, position_ids=position_ids)[0]
            #     print('<--------- FORWARD PROFILE --------->')
            #     print(p.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            # else:
            out = module(x, position_ids=position_ids)[0]
            if fsdp:
                dist.barrier()
        torch.cuda.synchronize()
        if fsdp:
            dist.barrier()

        end_time = time.perf_counter()

        return (end_time - start_time) * 1000 / num_bench_steps


def get_llama2_7b(torch_dtype):
    model = transformers.LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf',
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


CONFIGS = {
    'base': {
        'quant_config.fwd.had_x': 'none',
        'quant_config.fwd.had_w': 'none',
        'quant_config.bwd1.had_e': 'none',
        'quant_config.bwd1.had_w': 'none',
        'quant_config.bwd2.had_e': 'none',
        'quant_config.bwd2.had_x': 'none',
        'quant_config.fwd.quant_x': 'none',
        'quant_config.fwd.quant_w': 'none',
        'quant_config.bwd1.quant_e': 'none',
        'quant_config.bwd1.quant_w': 'none',
        'quant_config.bwd2.quant_e': 'none',
        'quant_config.bwd2.quant_x': 'none',
        'quant_config.kernel': 'none',
        'quant_config.simulate': True,
        # 'quant_config.quantized_rms': 'none',
    },
    'mlsys-int8-global': {
        'quant_config.fwd.had_x': 'right',
        'quant_config.fwd.had_w': 'right',
        'quant_config.bwd1.had_e': 'left',
        'quant_config.bwd1.had_w': 'none',
        'quant_config.bwd2.had_e': 'none',
        'quant_config.bwd2.had_x': 'right',
        'quant_config.fwd.quant_x': 'global',
        'quant_config.fwd.quant_w': 'global',
        'quant_config.bwd1.quant_e': 'global',
        'quant_config.bwd1.quant_w': 'global',
        'quant_config.bwd2.quant_e': 'global',
        'quant_config.bwd2.quant_x': 'global',
    },
    'mlsys-int8-structured': {
        'quant_config.fwd.had_x': 'right',
        'quant_config.fwd.had_w': 'right',
        'quant_config.bwd1.had_e': 'left',
        'quant_config.bwd1.had_w': 'none',
        'quant_config.bwd2.had_e': 'none',
        'quant_config.bwd2.had_x': 'right',
        'quant_config.fwd.quant_x': 'row_wise',
        'quant_config.fwd.quant_w': 'row_wise',
        'quant_config.bwd1.quant_e': 'row_wise',
        'quant_config.bwd1.quant_w': 'col_wise',
        'quant_config.bwd2.quant_e': 'col_wise',
        'quant_config.bwd2.quant_x': 'col_wise',
    },
    'no-had-fp8-global': {
        'quant_config.fwd.had_x': 'none',
        'quant_config.fwd.had_w': 'none',
        'quant_config.bwd1.had_e': 'none',
        'quant_config.bwd1.had_w': 'none',
        'quant_config.bwd2.had_e': 'none',
        'quant_config.bwd2.had_x': 'none',
        'quant_config.fwd.quant_x': 'fp8_e4m3fnuz',
        'quant_config.fwd.quant_w': 'fp8_e4m3fnuz',
        'quant_config.bwd1.quant_e': 'fp8_e4m3fnuz',
        'quant_config.bwd1.quant_w': 'fp8_e4m3fnuz',
        'quant_config.bwd2.quant_e': 'fp8_e4m3fnuz',
        'quant_config.bwd2.quant_x': 'fp8_e4m3fnuz',
    },
    'bwd1-eh-fp8-global': {
        'quant_config.fwd.had_x': 'none',
        'quant_config.fwd.had_w': 'none',
        'quant_config.bwd1.had_e': 'right',
        'quant_config.bwd1.had_w': 'left',
        'quant_config.bwd2.had_e': 'none',
        'quant_config.bwd2.had_x': 'none',
        'quant_config.fwd.quant_x': 'fp8_e4m3fnuz',
        'quant_config.fwd.quant_w': 'fp8_e4m3fnuz',
        'quant_config.bwd1.quant_e': 'fp8_e4m3fnuz',
        'quant_config.bwd1.quant_w': 'fp8_e4m3fnuz',
        'quant_config.bwd2.quant_e': 'fp8_e4m3fnuz',
        'quant_config.bwd2.quant_x': 'fp8_e4m3fnuz',
    },
    'bwd1-he-fp8-global': {
        'quant_config.fwd.had_x': 'none',
        'quant_config.fwd.had_w': 'none',
        'quant_config.bwd1.had_e': 'left',
        'quant_config.bwd1.had_w': 'none',
        'quant_config.bwd2.had_e': 'none',
        'quant_config.bwd2.had_x': 'none',
        'quant_config.fwd.quant_x': 'fp8_e4m3fnuz',
        'quant_config.fwd.quant_w': 'fp8_e4m3fnuz',
        'quant_config.bwd1.quant_e': 'fp8_e4m3fnuz',
        'quant_config.bwd1.quant_w': 'fp8_e4m3fnuz',
        'quant_config.bwd2.quant_e': 'fp8_e4m3fnuz',
        'quant_config.bwd2.quant_x': 'fp8_e4m3fnuz',
    },
    'no-had-just-fwd-int8-global': {
        'quant_config.fwd.had_x': 'none',
        'quant_config.fwd.had_w': 'none',
        'quant_config.bwd1.had_e': 'none',
        'quant_config.bwd1.had_w': 'none',
        'quant_config.bwd2.had_e': 'none',
        'quant_config.bwd2.had_x': 'none',
        'quant_config.fwd.quant_x': 'global',
        'quant_config.fwd.quant_w': 'global',
        'quant_config.bwd1.quant_e': 'none',
        'quant_config.bwd1.quant_w': 'none',
        'quant_config.bwd2.quant_e': 'none',
        'quant_config.bwd2.quant_x': 'none',
    },
    'switchback': {
        'quant_config.fwd.had_x': 'none',
        'quant_config.fwd.had_w': 'none',
        'quant_config.bwd1.had_e': 'none',
        'quant_config.bwd1.had_w': 'none',
        'quant_config.bwd2.had_e': 'none',
        'quant_config.bwd2.had_x': 'none',
        'quant_config.fwd.quant_x': 'row_wise',
        'quant_config.fwd.quant_w': 'global',
        'quant_config.bwd1.quant_e': 'row_wise',
        'quant_config.bwd1.quant_w': 'global',
        'quant_config.bwd2.quant_e': 'none',
        'quant_config.bwd2.quant_x': 'none',
        'quant_config.quantized_rms': 'fp8_e4m3fn',
    },
    'fp8_pure': {
        'quant_config.fwd.had_x': 'none',
        'quant_config.fwd.had_w': 'none',
        'quant_config.bwd1.had_e': 'none',
        'quant_config.bwd1.had_w': 'none',
        'quant_config.bwd2.had_e': 'none',
        'quant_config.bwd2.had_x': 'none',
        'quant_config.fwd.quant_x': 'none',
        'quant_config.fwd.quant_w': 'none',
        'quant_config.bwd1.quant_e': 'none',
        'quant_config.bwd1.quant_w': 'none',
        'quant_config.bwd2.quant_e': 'none',
        'quant_config.bwd2.quant_x': 'none',
        'quant_config.kernel': 'fp8_pure',
        'quant_config.simulate': True,
        # 'quant_config.quantized_rms': 'none',
    },
    'fp8_hey': {
        'quant_config.fwd.had_x': 'none',
        'quant_config.fwd.had_w': 'none',
        'quant_config.bwd1.had_e': 'none',
        'quant_config.bwd1.had_w': 'none',
        'quant_config.bwd2.had_e': 'none',
        'quant_config.bwd2.had_x': 'none',
        'quant_config.fwd.quant_x': 'none',
        'quant_config.fwd.quant_w': 'none',
        'quant_config.bwd1.quant_e': 'none',
        'quant_config.bwd1.quant_w': 'none',
        'quant_config.bwd2.quant_e': 'none',
        'quant_config.bwd2.quant_x': 'none',
        'quant_config.kernel': 'fp8_hey',
        'quant_config.simulate': True,
        # 'quant_config.quantized_rms': 'none',
    }
}


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


def run_fwd_bwd(module, x, position_ids):
    x_cpy = x.clone()
    x_cpy.requires_grad_()
    x_cpy.retain_grad()
    setattr(x_cpy, 'grad', None)
    module.zero_grad()
    out = module(x_cpy, position_ids=position_ids)[0]
    out.sum().backward()
    return out.detach().cpu(), torch.cat(
        [param.grad.clone().view(-1) for param in module.parameters()]).detach().cpu(), x_cpy.grad.detach().cpu()


def error(x, y):
    return torch.norm(x - y).item() / torch.norm(y).item()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Argument parser for model parameters and benchmark settings.")

    parser.add_argument('--seq_len',
                        type=int,
                        default=512,
                        help='Sequence length.')

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

    parser.add_argument('--hq_schemes', nargs='+', help='HQ schemes to use.', default=['base'])

    parser.add_argument('--include_fsdp_qcomm',
                        default=False,
                        action='store_true',
                        help='Whether to include the FSDP quantized communication version.')

    parser.add_argument('--include_base_fsdp_deep_wrap',
                        default=False,
                        action='store_true',
                        help='Whether to include the FSDP deeper wrap version for base.')

    args = parser.parse_args()

    bits = 8
    bsz_list = [8]
    # bsz_list = [4, 8, 16, 32]
    dtype = 'bf16'

    assert dtype in ['fp16', 'bf16']
    torch_dtype = torch.float16 if dtype == 'fp16' else torch.bfloat16

    seq_len = args.seq_len

    baseline_model_org = prepare_num_blocks(
        get_llama2_7b(torch_dtype),
        num_blocks=args.num_blocks,
        keep_lm_head=args.keep_lm_head
    )

    modules = []
    names = []
    module_args = []

    for cfg_name in args.hq_schemes:
        cfg = CONFIGS[cfg_name]
        hq_config = {'fwd': {}, 'bwd1': {}, 'bwd2': {}, 'kernel': 'none', 'simulate': True}
        for k, v in cfg.items():
            parts = k.split('.')
            if len(parts) == 3:
                hq_config[parts[1]][parts[2]] = v
            elif len(parts) == 2:
                hq_config[parts[1]] = v
            else:
                raise ValueError(f'Invalid config key: {k}')
        print(hq_config)

        model = wrap_model(copy.deepcopy(baseline_model_org), hq_config)

        modules.append(model)
        module_args.append({'fsdp_qcomm': False, 'fsdp_deep_wrap': False})
        names.append(cfg_name)

        if cfg_name == 'base' and args.include_base_fsdp_deep_wrap:
            assert args.fsdp
            model = wrap_model(copy.deepcopy(baseline_model_org), hq_config)
            modules.append(model)
            module_args.append({'fsdp_qcomm': False, 'fsdp_deep_wrap': True})
            names.append(f'{cfg_name} (deep fsdp wrap)')

        if args.include_fsdp_qcomm:
            assert args.fsdp
            model = wrap_model(copy.deepcopy(baseline_model_org), hq_config)
            modules.append(model)
            module_args.append({'fsdp_qcomm': True, 'fsdp_deep_wrap': True})
            names.append(f'{cfg_name} (qcomm fsdp)')

    if args.fsdp:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist_setup(rank, world_size)


        def auto_wrap_policy(module, recurse, nonwrapped_numel):
            return isinstance(module, LlamaDecoderLayer)


        def deep_auto_wrap_policy(module, recurse, nonwrapped_numel):
            return isinstance(module, LlamaSdpaAttention) or isinstance(module, LlamaMLP)


        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
        torch.cuda.set_device(rank)

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        for i, module in enumerate(modules):
            qcomm = module_args[i]['fsdp_qcomm']
            deep_wrap = module_args[i]['fsdp_deep_wrap']

            # if deep_wrap:
            #     for n, m in module.named_modules():
            #         if isinstance(m, LlamaMLP) or isinstance(m, LlamaSdpaAttention):
            #             fsdp_m = FSDP(
            #                 m,
            #                 auto_wrap_policy=None,
            #                 mixed_precision=mp_policy,
            #                 forward_prefetch=True,
            #                 device_id=torch.cuda.current_device()
            #             )
            #             swap_module(module, n, fsdp_m)
            #
            # modules[i] = FSDP(
            #     module,
            #     auto_wrap_policy=None if deep_wrap else auto_wrap_policy,
            #     mixed_precision=mp_policy,
            #     forward_prefetch=True,
            #     device_id=torch.cuda.current_device()
            # )

            ignored_modules = [m for m in modules[i].modules() if isinstance(m, LlamaRMSNorm)] if deep_wrap else None
            if ignored_modules is not None:
                for m in ignored_modules:
                    m.to(device=torch.cuda.current_device())
            print(ignored_modules)
            modules[i] = FSDP(
                module,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mp_policy,
                forward_prefetch=True,
                device_id=torch.cuda.current_device(),
                ignored_modules=ignored_modules
            )

            if qcomm:
                qllmt.nn.patch_fsdp_model(modules[i], qdtype=torch.float8_e4m3fn)  # fixme, make dtype configurable
            print(modules[i])
    else:
        for module in modules:
            module.cuda()

    rank = dist.get_rank() if args.fsdp else 0
    world_size = dist.get_world_size() if args.fsdp else 1
    device_name = torch.cuda.get_device_name().split()[-1]

    speedups_all = [[] for _ in modules]
    speedups_fwd = [[] for _ in modules]
    speedups_bwd = [[] for _ in modules]

    for bsz in bsz_list:
        # inputs
        x_org = torch.randn(bsz, seq_len, baseline_model_org.config.hidden_size, requires_grad=True,
                            dtype=torch_dtype).cuda(rank)
        xs = [torch.empty_like(x_org).copy_(x_org).to(torch_dtype).cuda(rank) for _ in modules]
        for x in xs:
            x.requires_grad_()

        position_ids = torch.arange(seq_len).unsqueeze(0).cuda()

        times_all = [llama_decoder_benchmark(module, x, position_ids, fsdp=args.fsdp) for x, module in zip(xs, modules)]

        times_fwd = [llama_decoder_benchmark_fwd(module, x, position_ids, fsdp=args.fsdp) for x, module in
                     zip(xs, modules)]

        outs_grads = [run_fwd_bwd(module, x, position_ids=position_ids) for x, module in zip(xs, modules)]
        outs = [out for out, _, _ in outs_grads]
        grad_ws = [grad_w for _, grad_w, _ in outs_grads]
        grad_xs = [grad_x for _, _, grad_x in outs_grads]

        # if dist.get_rank() == 0:
        #     # print(outs)
        #     print(grad_ws[0])
        #     print(grad_ws[1])
        #     print(grad_ws[2])

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

        # del baseline_model, had_model
        del x, xs, position_ids
        gc.collect()
    if args.fsdp:
        dist.barrier()
        dist_cleanup()

    # if rank == 0:
    #     def theme():
    #         import matplotlib
    #         # Apply the default theme
    #         sns.set_theme()
    #         sns.set(font_scale=1.3)
    #         sns.set_style("darkgrid", {"axes.facecolor": ".95"})
    #         # set fonttype
    #         matplotlib.rcParams['pdf.fonttype'] = 42
    #         matplotlib.rcParams['ps.fonttype']  = 42
    #         matplotlib.rcParams['font.family'] = 'serif'

    #     file_name = f'{device_name}_llama_decoder_benchmark'
    #     title = f'LLaMa2-7B Decoder (seq_len={args.seq_len}) on {device_name}'
    #     # for name, spfwd, spbwd, spall in zip(names[1:], speedups_fwd[1:], speedups_bwd[1:], speedups_all[1:]):
    #     #     plt.plot(bsz_list, spfwd, '-o', label=f'{name}-fwd')
    #     #     plt.plot(bsz_list, spbwd, '-o', label=f'{name}-bwd')
    #     #     plt.plot(bsz_list, spall, '-o', label=f'{name}-fwd+bwd')

    #     #     plt.axhline(1, color='black', linestyle='--')
    #     #     plt.xlabel('Batch size')
    #     #     plt.ylabel(f'Speedup (over {dtype.upper()})')
    #     #     plt.legend()
    #     #     plt.title(title)
    #     #     plt.savefig(f'{file_name}.png')

    #     theme()
    #     for name, spfwd in zip(names[1:], speedups_fwd[1:]):
    #         plt.plot(bsz_list, spfwd, '-o', label=f'{name}')

    #     plt.axhline(1, color='black', linestyle='--')
    #     plt.xlabel('Batch size')
    #     plt.ylabel(f'Speedup (over {dtype.upper()})')
    #     plt.legend()
    #     # plt.title(f'{title} - Forward')
    #     plt.savefig(f'{file_name}_fwd.pdf')
    #     plt.close()

    #     theme()
    #     for name, spbwd in zip(names[1:], speedups_bwd[1:]):
    #         plt.plot(bsz_list, spbwd, '-o', label=f'{name}')

    #     plt.axhline(1, color='black', linestyle='--')
    #     plt.xlabel('Batch size')
    #     plt.ylabel(f'Speedup (over {dtype.upper()})')
    #     plt.legend()
    #     # plt.title(f'{title} - Backward')
    #     plt.savefig(f'{file_name}_bwd.pdf')
    #     plt.close()

    #     theme()
    #     for name, spall in zip(names[1:], speedups_all[1:]):
    #         plt.plot(bsz_list, spall, '-o', label=f'{name}')

    #     plt.axhline(1, color='black', linestyle='--')
    #     plt.xlabel('Batch Size')
    #     plt.ylabel(f'Speedup (over {dtype.upper()})')
    #     plt.legend()
    #     # plt.title(f'{title} - Fwd+Bwd')
    #     plt.savefig(f'{file_name}_all.pdf')
    #     plt.close()
