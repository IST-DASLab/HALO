import torch, time, gc, tqdm, qllmt, math
import transformers
import numpy as np
import argparse
import copy
import datasets
from qllmt.nn import wrap_llama
from torch.utils.data import DataLoader, Dataset
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.set_printoptions(precision=4)

wikitext_ppl =  {
    'meta-llama/Llama-2-7b-hf': 5.47,
    'TinyLlama/TinyLlama_v1.1': 7.71
}

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass


def get_llama(model_name, hf_token, dtype_='auto'):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers. AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype_,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    print('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model

def prepare_test_dataloader(
    dataset: datasets.Dataset, 
    tokenizer,
    seqlen: int = 2048, 
    batch_size: int = 1
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a test dataset. This dataloader should be used when comparing WikiText2 perplexities with other papers, e.g. SparseGPT (arxiv.org/abs/2301.00774).

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        seqlen: The sequence length of sequences in the dataset.
        batch_size: The batch size.

    Returns:
        A DataLoader.
    """

    print(f"Preparing test dataloader", end=" ---> ")

    class TestDataset(Dataset):
        def __init__(self, ds, tokenizer, seqlen=2048):
            """Tokenize the entire dataset and reshape it into sequences of length seqlen."""
            try:
                tokenized_ds = tokenizer("\n\n".join(ds['text']), return_tensors='pt')
            except KeyError:
                tokenized_ds = tokenizer("\n\n".join(ds['sentence']), return_tensors='pt')
            nsamples = tokenized_ds.input_ids.numel() // seqlen

            input_ids = tokenized_ds.input_ids[0, : nsamples * seqlen]
            input_ids = input_ids.reshape(nsamples, seqlen)
            attn_mask = tokenized_ds.attention_mask[0, : nsamples * seqlen]
            attn_mask = attn_mask.reshape(nsamples, seqlen)

            self.input_ids = input_ids
            self.attn_mask = attn_mask

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attn_mask[idx]}

        def __len__(self):
            return len(self.input_ids)

    test_ds = TestDataset(dataset, tokenizer, seqlen)
    loader = DataLoader(test_ds, batch_size=batch_size)
    print(f"Preparing test dataloader done")
    return loader


@torch.no_grad()
def ppl_evaluator(
    model, 
    pad_token_id,
    testloader
) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    def map_tensors(obj,  device, dtype=None):
        """Recursively map tensors to device and dtype."""
        if isinstance(obj, torch.Tensor):
            if device is not None:
                obj = obj.to(device=device)
            if dtype is not None:
                obj = obj.to(dtype=dtype)
            return obj
        elif isinstance(obj, (list, tuple)):
            return type(obj)(map_tensors(x, device, dtype) for x in obj)
        elif isinstance(obj, dict):
            return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
        else:
            return obj

    start_time = time.time()

    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    for batch in tqdm.tqdm(testloader, desc="Evaluating perplexity", unit="batch"):
        batch = map_tensors(batch, 'cuda:0')
        
        logits = model(**batch).logits

        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())


    elapsed = time.time() - start_time
    print(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )
    
    return ppl.item()


def evaluator(model, tokenizer):

    ppl_based_tasks = ['wikitext2']

    for gen_task in ppl_based_tasks:
        generation_testloader = prepare_test_dataloader(
        dataset=datasets.load_dataset("wikitext", name="wikitext-2-raw-v1", data_files=None)["test"], 
        tokenizer=tokenizer, 
        batch_size=2)
        ppl = ppl_evaluator(model.to('cuda:0'), tokenizer.pad_token_id, generation_testloader) 
        # print perplexity up to 2 decimal places
        print(f"{gen_task.upper()} Perplexity: {ppl:.2f}")


# define a hook in the backward pass for lm_head
baseline_grads = []
baseline_inputs = []
had_grads = []
had_inputs = []
def forward_baseline_hook(module, input, output):
    if type(input) == tuple:
        baseline_inputs.append(input[0].cpu())
        baseline_inputs.append(output[0].cpu())
    else:
        baseline_inputs.append(input.cpu())
        baseline_inputs.append(output.cpu())
def forward_had_hook(module, input, output):
    if type(input) == tuple:
        had_inputs.append(input[0].cpu())
        had_inputs.append(output[0].cpu())
    else:
        had_inputs.append(input.cpu())
        had_inputs.append(output.cpu())

# Define a backward hook function to store the gradients
def baseline_hook(module, grad_input, grad_output):
    # grad_output is a tuple, we are interested in the first element
    baseline_grads.append(grad_output[0].cpu())
    if grad_input[0] is not None:
        baseline_grads.append(grad_input[0].cpu())
def had_hook(module, grad_input, grad_output):
    # grad_output is a tuple, we are interested in the first element
    had_grads.append(grad_output[0].cpu()) #E_y
    if grad_input[0] is not None:
        had_grads.append(grad_input[0].cpu()) #E_x
        

def register_baseline_hooks(module):
    # forward hooks
    module.register_forward_hook(forward_baseline_hook)
    # backward hooks
    module.register_full_backward_hook(baseline_hook)
    
def register_had_hooks(had_module):
    had_module.register_forward_hook(forward_had_hook)
    had_module.register_full_backward_hook(had_hook)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Argument parser for model parameters and benchmark settings.")

    parser.add_argument('--model', 
                        type=str, 
                        default='TinyLlama/TinyLlama_v1.1', 
                        help='LLaMa Model (Default: TinyLLaMa)',
                        choices=['meta-llama/Llama-2-7b-hf', 'TinyLlama/TinyLlama_v1.1'])
    parser.add_argument('--seq_len', 
                        type=int, 
                        default=512, 
                        help='Sequence length.')
    parser.add_argument('--bits', 
                        type=int, 
                        default=16, 
                        choices=[16, 8],
                        help='Precision (default:16)')
    args = parser.parse_args()

    dtype_ = torch.float32

    baseline_model = get_llama(args.model, None, dtype_=dtype_)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model,
                                               use_fast=True,
                                               token=None)
    had_model = wrap_llama(copy.deepcopy(baseline_model), 
                           g_had=False, bitwidth=args.bits).to(dtype_)
    if args.bits == 16:
        print(f'WikiText PPL should be around {wikitext_ppl[args.model]}')

    if '7b' in args.model: 
        print('----> 7B model only Does Forward Pass')
        baseline_model.eval()
        with torch.no_grad():
            evaluator(had_model, tokenizer)  
    else:
        # with torch.no_grad():
        #     evaluator(had_model, tokenizer)
        baseline_model.train()
        had_model.train()
        had_model = had_model.cuda()
        baseline_model = baseline_model.cuda()


        # Register forward and backward hooks:

        # heads
        # register_baseline_hooks(baseline_model.lm_head)
        # register_had_hooks(had_model.lm_head)

        
        layer_idx = 21
        register_baseline_hooks(baseline_model.model.layers[layer_idx].mlp.down_proj)
        register_had_hooks(had_model.model.layers[layer_idx].mlp.module.down_proj)


        # Embedding 
        # register_baseline_hooks(baseline_model.model.embed_tokens)
        # register_had_hooks(had_model.model.embed_tokens)

        # last norm
        # register_baseline_hooks(baseline_model.model.norm)
        # register_had_hooks(had_model.model.norm)

        # define input and run forward and backward passes
        x = torch.randint(300, 2000, (1, args.seq_len), dtype=torch.long).cuda()
        x_had = torch.empty_like(x).copy_(x).detach_().cuda()
        y_baseline = baseline_model(x)
        y_had = had_model(x_had)
        y_baseline[0].backward(torch.ones_like(y_baseline[0])/100, retain_graph=True)
        y_had[0].backward(torch.ones_like(y_had[0])/100, retain_graph=True)


        # print diffs
        hooked_x = baseline_inputs[0]
        hooked_had_x = had_inputs[0]

        hooked_y = baseline_inputs[1]
        hooked_had_y = had_inputs[1]

        hooked_E_y = baseline_grads[0]
        hooked_had_E_y = had_grads[0]

        hooked_E_x = baseline_grads[1]
        hooked_had_E_x = had_grads[1]

        print(f'X   diff:', torch.max(torch.abs(hooked_x - hooked_had_x)))
        print(f'Y   diff:', torch.max(torch.abs(hooked_y - hooked_had_y)))
        print(f'E_y diff:', torch.max(torch.abs(hooked_E_y - hooked_had_E_y)))
        print(f'E_x diff:', torch.max(torch.abs(hooked_E_x - hooked_had_E_x)))

        print('\n\nG_w:')
        # print the names of the modules and their shapes in both models
        for fused_named_param, had_named_params in zip(baseline_model.named_parameters(), had_model.named_parameters()):
            fused_module = fused_named_param[-1]
            had_module = had_named_params[-1]
            print(fused_named_param[0], fused_module.shape, torch.max(torch.abs(fused_module.grad - had_module.grad)))

        print('-------')