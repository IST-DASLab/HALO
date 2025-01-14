import torch, time, gc
import numpy as np
import argparse
import copy
from qllmt.nn import wrap_jetfire_gpt2

import matplotlib.pyplot as plt

num_warmup_steps = 3
num_bench_steps = 9

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass


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
def gpt2_benchmark(model, x):
    # warmup
    for i in range(num_warmup_steps):
        out = model(x)[0]
        out.sum().backward()
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = model(x)[0]
        out.sum().backward()
    torch.cuda.synchronize()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps



def get_gpt2(model_name, hf_token=None):
    import qllmt
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = qllmt.nn.JetfireGPT.from_pretrained(model_name)
    print('---> Loading {} Model (JetFire Paper)'.format(model_name))
    print(model.config)
    return model



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Argument parser for model parameters and benchmark settings.")

    parser.add_argument('--model', 
                        default='gpt2-medium',
                        choices=['gpt2-medium', 'gpt2', 'gpt2-large'],#'gpt2-xl' is not supported: we do not have Hadamard for 4800 dim.
                        help='Model name.')
    parser.add_argument('--seq_len', 
                        type=int, 
                        default=512, 
                        help='Sequence length.')
    args = parser.parse_args()

    bits = 8
    bsz_list = [4, 8, 16, 32]
    seq_len = args.seq_len
    int8_speedups = []
    int8_grouped_speedups = []


    baseline_model = get_gpt2(args.model, None).half()
    had_model = wrap_jetfire_gpt2(copy.deepcopy(baseline_model), g_had=False, bitwidth=bits).half()
    had_model_group = wrap_jetfire_gpt2(copy.deepcopy(baseline_model), g_had=True, bitwidth=bits).half()
    baseline_model.config.use_cache = False
    

    for bsz in bsz_list:   

        #inputs
        x = torch.randint(0, baseline_model.config.vocab_size, (bsz, seq_len)).cuda()
        x_had = x.clone().cuda()

        fp16_time = gpt2_benchmark(baseline_model.cuda(), x)
        int8_time = gpt2_benchmark(had_model.cuda(), x)
        int8_grouped_had = gpt2_benchmark(had_model_group.cuda(), x)
        print('-----------------')
        print(f"Batch size: {bsz}, Sequence length: {seq_len}")
        print(f"FP16 time: {np.mean(fp16_time):.3f} +- {1.96 * np.std(fp16_time):.3f}ms")
        print(f"INT8 time: {np.mean(int8_time):.3f} +- {1.96 * np.std(int8_time):.3f}ms")
        print(f"INT8 (grouped) time: {np.mean(int8_grouped_had):.3f} +- {1.96 * np.std(int8_grouped_had):.3f}ms")
        int8_speedups.append(np.mean(fp16_time) / np.mean(int8_time))
        int8_grouped_speedups.append(np.mean(fp16_time) / np.mean(int8_grouped_had))

        del x, x_had
        gc.collect()

    plt.plot(bsz_list, int8_speedups, '-o', label='INT8')
    plt.plot(bsz_list, int8_grouped_speedups, '-o', label='INT8 (grouped)')
    plt.axhline(1, color='black', linestyle='--')
    plt.xlabel('Batch size')
    plt.ylabel('Speedup (over FP16)')
    plt.legend()
    plt.title(f'LLaMa2-7B Decoder (seq_len={args.seq_len}) on {torch.cuda.get_device_name()}')
    plt.savefig(f'{torch.cuda.get_device_name()}_{args.model}_benchmark.png')