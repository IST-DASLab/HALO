import torch, time, gc
import transformers
import numpy as np
import argparse
import copy
from qllmt.nn import wrap_model
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns

num_warmup_steps = 3
num_bench_steps = 30

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
def llama_decoder_benchmark(module, x, position_ids, checkpoint=False):
    # warmup
    for i in range(num_warmup_steps):
        if checkpoint:
            out = torch.utils.checkpoint.checkpoint(partial(
                module, 
                position_ids=position_ids
            ), x, use_reentrant=True)[0]
        else:
            out = module(x, position_ids=position_ids)[0]
        out.sum().backward()
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        if checkpoint:
            out = torch.utils.checkpoint.checkpoint(partial(
                module, 
                position_ids=position_ids
            ), x, use_reentrant=True)[0]
        else:
            out = module(x, position_ids=position_ids)[0]
        out.sum().backward()
    torch.cuda.synchronize()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps


@repeated_run()
def llama_decoder_benchmark_fwd(module, x, position_ids, checkpoint=False):
    with torch.no_grad():
        # warmup
        for i in range(num_warmup_steps):
            if checkpoint:
                out = torch.utils.checkpoint.checkpoint(partial(
                    module, 
                    position_ids=position_ids
                ), x, use_reentrant=True)[0]
            else:
                out = module(x, position_ids=position_ids)[0]
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        for i in range(num_bench_steps):
            if checkpoint:
                out = torch.utils.checkpoint.checkpoint(partial(
                    module, 
                    position_ids=position_ids
                ), x, use_reentrant=True)[0]
            else:
                out = module(x, position_ids=position_ids)[0]
        torch.cuda.synchronize()

        end_time = time.perf_counter()

        return (end_time - start_time) * 1000 / num_bench_steps


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          attn_implementation="flash_attention_2",
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    print('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model

def get_llama2_7b(torch_dtype):
    model = transformers.LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf',
                                                          use_cache=False,
                                                          torch_dtype=torch_dtype,
                                                          attn_implementation="flash_attention_2",
                                                          trust_remote_code=True,)
    return model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Argument parser for model parameters and benchmark settings.")

    parser.add_argument('--seq_len', 
                        type=int, 
                        default=512, 
                        help='Sequence length.')
    
    parser.add_argument('--checkpoint', 
                        default=False, 
                        action='store_true',
                        help='Activation checkpointing.')
    
    args = parser.parse_args()

    bits = 8
    bsz_list = [4, 8, 16]
    dtype = 'bf16'

    assert dtype in ['fp16', 'bf16']
    torch_dtype = torch.float16 if dtype == 'fp16' else torch.bfloat16

    seq_len = args.seq_len
    int8_speedups_all = []
    int8_speedups_fwd = []
    int8_speedups_bwd = []

    baseline_model = get_llama2_7b(torch_dtype)
    hq_config = {
        'fwd': {
            'had_x': 'none',
            'had_w': 'none',
            'quant_x': 'none',
            'quant_w': 'none'
        },
        'bwd1': {
            'had_e': 'none',
            'had_w': 'none',
            'quant_e': 'none',
            'quant_w': 'none'
        },
        'bwd2': {
            'had_e': 'none',
            'had_x': 'none',
            'quant_e': 'none',
            'quant_x': 'none'
        }
    }
    had_model = wrap_model(copy.deepcopy(baseline_model), hq_config)

    for bsz in bsz_list:   
        #blocks
        baseline_block = baseline_model.model.layers[0].cuda()
        had_block = had_model.model.layers[0].cuda()

        #inputs
        x = torch.rand(bsz, seq_len, baseline_model.config.hidden_size, requires_grad=True, dtype=torch_dtype).cuda()
        x_had = torch.empty_like(x).copy_(x).to(torch_dtype).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).cuda()

        fp16_time_all = llama_decoder_benchmark(baseline_block, x, position_ids, checkpoint=args.checkpoint)
        int8_time_all = llama_decoder_benchmark(had_block, x_had, position_ids, checkpoint=args.checkpoint)

        fp16_time_fwd = llama_decoder_benchmark_fwd(baseline_block, x, position_ids, checkpoint=args.checkpoint)
        int8_time_fwd = llama_decoder_benchmark_fwd(had_block, x_had, position_ids, checkpoint=args.checkpoint)

        fp16_time_bwd = np.array(fp16_time_all) - np.array(fp16_time_fwd)
        int8_time_bwd = np.array(int8_time_all) - np.array(int8_time_fwd)
        
        print('-----------------')
        print(f"Batch size: {bsz}, Sequence length: {seq_len}")

        print(f"FP16 time (fwd): {np.mean(fp16_time_fwd):.3f} +- {1.96 * np.std(fp16_time_fwd):.3f}ms")
        print(f"INT8 time (fwd): {np.mean(int8_time_fwd):.3f} +- {1.96 * np.std(int8_time_fwd):.3f}ms    --> {np.mean(fp16_time_fwd) / np.mean(int8_time_fwd):.2f}x\n")

        print(f"FP16 time (bwd): {np.mean(fp16_time_bwd):.3f} +- {1.96 * np.std(fp16_time_bwd):.3f}ms")
        print(f"INT8 time (bwd): {np.mean(int8_time_bwd):.3f} +- {1.96 * np.std(int8_time_bwd):.3f}ms    --> {np.mean(fp16_time_fwd) / np.mean(int8_time_fwd):.2f}x\n")

        print(f"FP16 time (all): {np.mean(fp16_time_all):.3f} +- {1.96 * np.std(fp16_time_all):.3f}ms")
        print(f"INT8 time (all): {np.mean(int8_time_all):.3f} +- {1.96 * np.std(int8_time_all):.3f}ms    --> {np.mean(fp16_time_fwd) / np.mean(int8_time_fwd):.2f}x\n")


        int8_speedups_all.append(np.mean(fp16_time_all) / np.mean(int8_time_all))
        int8_speedups_fwd.append(np.mean(fp16_time_fwd) / np.mean(int8_time_fwd))
        int8_speedups_bwd.append(np.mean(fp16_time_bwd) / np.mean(int8_time_bwd))

        del baseline_block, had_block
        del x, x_had, position_ids
        gc.collect()

    plt.plot(bsz_list, int8_speedups_fwd, '-o', label='Fwd')
    plt.plot(bsz_list, int8_speedups_bwd, '-o', label='Bwd')
    plt.plot(bsz_list, int8_speedups_all, '-o', label='Fwd+Bwd')
    
    plt.axhline(1, color='black', linestyle='--')
    plt.xlabel('Batch size')
    plt.ylabel('Speedup (over FP16)')
    plt.legend()
    plt.title(f'LLaMa2-7B Decoder (seq_len={args.seq_len}) on {torch.cuda.get_device_name()}')
    # plt.savefig(f'{torch.cuda.get_device_name()}_llama_decoder_benchmark.png')
