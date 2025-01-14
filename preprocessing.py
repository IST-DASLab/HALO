import os
from typing import Dict

import transformers

# LLAMA2_PROMPT_START_WRAPPER = "<s>[INST] "
# LLAMA2_PROMPT_END_WRAPPER = " [/INST]"
# LLAMA2_RESPONSE_START_WRAPPER = ""
# LLAMA2_RESPONSE_END_WRAPPER = "</s>"
#
# LLAMA3_PROMPT_START_WRAPPER = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
# LLAMA3_PROMPT_END_WRAPPER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
# LLAMA3_RESPONSE_START_WRAPPER = ""
# LLAMA3_RESPONSE_END_WRAPPER = "<|eot_id|>"
#
#
# def get_model_special_tokens(model_name):
#     model_name = model_name.lower()
#     if "llama-3" in model_name:
#         prompt_start_wrapper = LLAMA3_PROMPT_START_WRAPPER
#         prompt_end_wrapper = LLAMA3_PROMPT_END_WRAPPER
#         response_start_wrapper = LLAMA3_RESPONSE_START_WRAPPER
#         response_end_wrapper = LLAMA3_RESPONSE_END_WRAPPER
#     elif "llama-2" in model_name.lower():
#         prompt_start_wrapper = LLAMA2_PROMPT_START_WRAPPER
#         prompt_end_wrapper = LLAMA2_PROMPT_END_WRAPPER
#         response_start_wrapper = LLAMA2_RESPONSE_START_WRAPPER
#         response_end_wrapper = LLAMA2_RESPONSE_END_WRAPPER
#     else:
#         raise ValueError(f"Presets missing for prompting model {model_name}")
#
#     return prompt_start_wrapper, prompt_end_wrapper, response_start_wrapper, response_end_wrapper
#
#
# model_name = os.environ.get("MODEL", "")
# if not model_name:
#     print(
#         "preprocessing.py: Warning: MODEL environment variable not set. Defaulting to 'llama-2' for preprocessor...")
# PROMPT_START_WRAPPER, PROMPT_END_WRAPPER, RESPONSE_START_WRAPPER, RESPONSE_END_WRAPPER = get_model_special_tokens(
#     model_name or "llama-2")


tokenizer_cache = {}


def get_tokenizer(model_name=None):
    if model_name is None:
        model_name = os.environ.get("MODEL", "")
    assert model_name, "either pass model_name or set MODEL environment variable"
    if model_name not in tokenizer_cache:
        tokenizer_cache[model_name] = transformers.AutoTokenizer.from_pretrained(model_name)
    return tokenizer_cache[model_name]


def chat_preprocess(text_prompt, text_response, tokenizer):
    prompt = tokenizer.apply_chat_template(
        conversation=[
            # {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": text_prompt},
            # {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_response = tokenizer.apply_chat_template(
        conversation=[
            # {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": text_prompt},
            {"role": "assistant", "content": text_response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    for i in range(len(prompt)):
        assert prompt[i] == prompt_response[i], f"Prompt mismatch at index {i}: {prompt[:i]} != {prompt_response[:i]}"
    response = prompt_response[len(prompt):]
    return {
        "prompt": prompt,
        "response": response,
    }


# ps_cache = {}
#
#
# def get_chat_response_prefix_suffix(tokenizer):
#     if tokenizer.name_or_path in ps_cache:
#         return ps_cache[tokenizer.name_or_path]
#     prompt = tokenizer.apply_chat_template(
#         conversation=[
#             {"role": "user", "content": "$"},
#         ],
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     prompt_response = tokenizer.apply_chat_template(
#         conversation=[
#             {"role": "user", "content": "$"},
#             {"role": "assistant", "content": "$"},
#         ],
#         tokenize=False,
#         add_generation_prompt=False,
#     )
#     for i in range(len(prompt)):
#         assert prompt[i] == prompt_response[i], f"Prompt mismatch at index {i}: {prompt[:i]} != {prompt_response[:i]}"
#     response = prompt_response[len(prompt):]
#     response_split = response.split("$")
#     assert len(response_split) == 2, f"Response split length is not 2: {response_split}"
#     ps_cache[tokenizer.name_or_path] = response_split
#     return response_split[0], response_split[1]


# def chat_response_strip(text, tokenizer=None, model_name=None):
#     if tokenizer is None:
#         tokenizer = get_tokenizer(model_name=model_name)
#     prefix, suffix = get_chat_response_prefix_suffix(tokenizer)
#
#     return text[len(prefix): -len(suffix)].strip()


def gsm8k_preprocessing_function(inp: Dict) -> Dict:
    try:
        if 'input' not in inp:
            inp['input'] = inp['question']
        if 'output' not in inp:
            inp['output'] = inp['answer']
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from {inp}") from e
    return chat_preprocess(inp['input'], inp['output'], get_tokenizer())


def sql_preprocessing_function(inp: Dict) -> Dict:
    """Split out prompt/response from text."""
    try:
        if 'input' not in inp:
            inp['input'] = inp['messages'][0]['content']
        if 'output' not in inp:
            inp['output'] = inp['messages'][1]['content']
    except Exception as e:
        raise ValueError(
            f"Unable to extract prompt/response from 'text'={inp['text']}"
        ) from e
    return chat_preprocess(inp['input'], inp['output'], get_tokenizer())


VIGGO_PROMPT = ''.join([
    "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. ",
    "This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. ",
    "The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
    "\n\n### Target sentence:\n{target}"
])


def viggo_preprocessing_function(inp: Dict) -> Dict:
    try:
        if 'input' not in inp:
            inp['input'] = inp['target']
        if 'output' not in inp:
            inp['output'] = inp['meaning_representation']
        prompt = VIGGO_PROMPT.format(target=inp['input'])
        response = inp['output']
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from {inp}") from e
    return chat_preprocess(prompt, response, get_tokenizer())


def code_preprocessing_function(inp: Dict) -> Dict:
    try:
        if 'input' not in inp:
            inp['input'] = inp['instruction']
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from {inp}") from e
    return chat_preprocess(inp['input'], inp['output'], get_tokenizer())
