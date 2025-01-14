import json
import os.path
import random

import pandas as pd
import fire
import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

import sys

from peft import AutoPeftModelForCausalLM

sys.path.append('..')
import preprocessing

WANDB_ENTITY = os.environ.get('WANDB_ENTITY', None)
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', None)

model_name = os.environ.get("MODEL", "")


def batch_eval(model, tokenizer, dataset_name, batch):
    preprocess_fn = getattr(preprocessing, f'{dataset_name}_preprocessing_function')
    chat_text = [preprocess_fn({'input': inp, 'output': ''})['prompt'] for inp in batch]

    model_inputs = tokenizer(chat_text, return_tensors="pt", padding=True).to("cuda")
    generated_ids = model.generate(
        **model_inputs,
        max_length=512, do_sample=False, num_beams=1, top_p=None, temperature=None,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id
    )
    results_raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    generated_ids = generated_ids[..., model_inputs['input_ids'].shape[-1]:]
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    results = [res.strip() for res in results]

    return results, results_raw


def is_correct(pred, label, dataset_name):
    pred, label = pred.strip(), label.strip()
    if dataset_name == 'gsm8k':
        return pred.split('####')[-1].strip() == label.split('####')[-1].strip()
    elif dataset_name == 'viggo':
        pred_fn = pred.split('(')[0]
        true_fn = label.split('(')[0]
        if pred_fn != true_fn:
            return False
        fn = true_fn
        pred_attrs = sorted([p.strip() for p in pred.replace(fn, '').strip('()').split(',')])
        label_attrs = sorted([l.strip() for l in label.replace(fn, '').strip('()').split(',')])
        return len(pred_attrs) == len(label_attrs) and all([p == l for p, l in zip(pred_attrs, label_attrs)])
    elif dataset_name == 'sql':
        return pred == label


@torch.no_grad()
def eval(model, tokenizer, dataset_name, subsample=None, batch_size=8):
    if dataset_name == 'sql':
        dataset = load_dataset('json',
                               data_files="../data/sql/valid.jsonl",
                               split="train")
        dataset = dataset.map(
            lambda example: {
                'inp': example['messages'][0]['content'],
                'label': example['messages'][1]['content'],
            }, remove_columns=['messages'])
    elif dataset_name == 'viggo':
        dataset = load_dataset('GEM/viggo', split='validation')
        dataset = dataset.map(
            lambda example: {
                'inp': example['target'],
                'label': example['meaning_representation']
            })
    elif dataset_name == 'gsm8k':
        dataset = load_dataset('gsm8k', 'main', split='test')
        dataset = dataset.map(
            lambda example: {
                'inp': example['question'],
                'label': example['answer']
            })
    else:
        assert False, f"Unknown dataset {dataset_name}"

    if subsample is not None:
        assert type(subsample) == int, "subsample must be an integer"
        assert subsample <= len(dataset), "subsample must be less than the dataset size"
        assert subsample > 0, "subsample must be greater than 0"
        dataset = dataset.select(random.sample(range(len(dataset)), subsample))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []
    results_raw = []
    labels_all = []
    for batch in tqdm(dataloader):
        preds, preds_raw = batch_eval(model, tokenizer, dataset_name, batch['inp'])
        labels = batch['label']
        results.extend([is_correct(pred, label, dataset_name) for pred, label in zip(preds, labels)])
        results_raw.extend(preds_raw)
        labels_all.extend(labels)
    return sum(results) / len(results), results_raw, results, labels_all


def load_model(model_path, precision):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from checkpointer import wrap_from_pretrained

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if precision in ['bf16', 'bfloat16'] else torch.float32
    if os.path.exists(os.path.join(model_path, 'adapter_config.json')):
        print('Loading PEFT model...')
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map='cpu',
            torch_dtype=dtype,
            trust_remote_code=True,
            use_auth_token=True,
            attn_implementation='sdpa',
            use_cache=False
        )
        print(model.config)
        print(model.peft_config)
        # print(model.model.base_model.layers[0].mlp.up_proj.base_layer.weight.dtype, "up_proj base_weight")
        # print(model.model.base_model.layers[0].mlp.up_proj.lora_A['default'].weight.dtype, "up_proj lora_A")
    else:
        print('Loading base model...')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cpu',
            torch_dtype=dtype,
            trust_remote_code=True,
            use_auth_token=True,
            attn_implementation='sdpa',
            use_cache=False
        )
        wrap_from_pretrained(model, model_path)

    model.eval()
    model.cuda()
    model.to(dtype=dtype)
    return model, tokenizer


def auto_batch_size():
    import torch
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if gpu_mem < 25:
        batch_size = 12
    elif gpu_mem < 50:
        batch_size = 32
    else:
        batch_size = 64
    return batch_size


def main(
        model_path,
        dataset,
        precision='bf16',
        subsample=None,
        force=False,
        write_out=True,
        batch_size='auto',
):
    if batch_size == 'auto':
        batch_size = auto_batch_size()
        print(f'Auto batch size: {batch_size}')
    out_path = os.path.join(model_path, f'eval.txt')
    if os.path.exists(out_path) and not force:
        print(f'evaluation already exists, skipping evaluation...')
    else:
        print(f'Evaluating {model_path} on {dataset}...')
        model, tokenizer = load_model(model_path, precision)
        acc, results_raw, results, labels = eval(model, tokenizer, dataset, subsample=subsample, batch_size=batch_size)
        with open(out_path, 'w') as f:
            metric = {f'{dataset}/validation_acc': acc}
            print(metric)
            json.dump(metric, f)

        if write_out:
            df = pd.DataFrame({'results_raw': results_raw, 'results': results, 'labels': labels})
            df.to_json(out_path.replace('.txt', '.jsonl'), orient='records', lines=True)

    if WANDB_PROJECT is not None:
        print(f'Updating wandb...')
        api = wandb.Api()
        run = api.runs(path=f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                       filters={"display_name": os.path.basename(model_path.rstrip('/'))})[0]
        with open(out_path, 'r') as f:
            metric = json.load(f)
        acc = metric[f'{dataset}/validation_acc']
        run.summary["eval_acc"] = acc
        run.summary.update()
        run.update()
    print('Done!')


if __name__ == '__main__':
    fire.Fire(main)
