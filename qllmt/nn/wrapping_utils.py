import torch

import qllmt
from qllmt.functional.hadamard import is_pow2
from .linear import SimulatedTwistFormerLinear
from .jetfire_sim import JetfireSimLinear

try:
    from .jetfire_linear import JetfireLinear
except ImportError:
    JetfireLinear = JetfireSimLinear

from .switchback import SwitchBackLinear
from .halo_linear import HaloLinear
from .fsdp import calculate_fake_tensor_info
from copy import deepcopy

# from ..functional.quantization import quantize_fp8


# try:
# except ImportError:
#     HuggingFaceModelWithFSDP = None
#     print("llmfoundry not found!")

class InputFwdHadamardWrapper(torch.nn.Module):
    def __init__(self,
                 module: torch.nn.Module
                 ):
        super().__init__()
        self.module = module

        if isinstance(module, torch.nn.Linear):
            assert is_pow2(module.in_features), 'Input features should be power of 2!'
        elif isinstance(module, torch.nn.Embedding):
            assert is_pow2(module.num_embeddings), 'Input features should be power of 2!'

    def forward(self, hidden_states, **kwargs):
        return self.module(
            qllmt.power_two_fwd_had(hidden_states, scale=1.0 / torch.tensor(hidden_states.shape[-1]).sqrt()),
            **kwargs)

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


def is_wrapped(model):
    for _, module in model.named_modules():
        if isinstance(module,
                      (InputFwdHadamardWrapper, SimulatedTwistFormerLinear, JetfireSimLinear, JetfireLinear, SwitchBackLinear, HaloLinear)):
            return True
    return False

def wrap_linear_module(module, config):
    kernel = config.get('kernel', 'simulated')
    if kernel == 'base':
        print(f'not wrapping since kernel is {kernel}')
        return module
    elif kernel.startswith('halo'):
        print(f'wrapping with {kernel}')
        return HaloLinear.from_unquantized(module, hq_config=config)
    elif kernel == 'jetfire_int8':
        print('wrapping with jetfire_int8')
        return JetfireSimLinear.from_unquantized(module, precision='int8')
    elif kernel == 'jetfire_fp6':
        print('wrapping with jetfire_fp6')
        return JetfireSimLinear.from_unquantized(module, precision='fp6')
    elif kernel == 'jetfire_real':
        print('wrapping with jetfire_real')
        return JetfireLinear.from_unquantized(module)
    elif kernel == 'switchback':
        print('wrapping with switchback')
        return SwitchBackLinear.from_unquantized(module)
    else:
        print('wrapping with twist')
        return SimulatedTwistFormerLinear.from_unquantized(module, hq_config=config)

def wrap_halo_qfsdp(model, config):
    kernel = config.get('kernel', 'simulated')
    assert 'halo' in kernel and 'qfsdp' in kernel, "kernel must be halo_qfsdp"

    try:
        block_cls = model.model.layers[0].__class__
        print(f'found block class', block_cls)
    except:
        raise ValueError('Unable to find the class for transformer blocks.')

    for block_name, block in model.named_modules():
        if isinstance(block, block_cls):
            block_linear_layers = [m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith(block_name + '.')]
            block_linear_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith(block_name + '.')]
            fake_tensors_info = calculate_fake_tensor_info(block_linear_layers)

            for i in range(len(block_linear_layers)):
                linear = block_linear_layers[i]
                name = block_linear_names[i]
                
                fake_before = 0
                fake_after = 0
                for fti in fake_tensors_info:
                    if fti['layer_idx'] == i and fti['side'] == 'before':
                        fake_before = fti['num_fake_params']
                    elif fti['layer_idx'] == i and fti['side'] == 'after':
                        fake_after = fti['num_fake_params']

                config_i = deepcopy(config)
                config_i['fake_before'] = fake_before
                config_i['fake_after'] = fake_after

                wrapped_linear = HaloLinear.from_unquantized(linear, config_i)
                swap_module(model, name, wrapped_linear)
            
            print(f"{block_name} wrapped")


def wrap_model(model, config, exceptions=['classifier', 'lm_head']):
    from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithFSDP
    orig_model = model
    if HuggingFaceModelWithFSDP is not None and isinstance(model, HuggingFaceModelWithFSDP):
        model = model.model

    assert config is not None, "config is required"
    setattr(model, "hadamard_config", config)
    print("wrapping config:", getattr(model, "hadamard_config"))

    assert not is_wrapped(model), "model is already wrapped"

    kernel = config.get('kernel', 'simulated')
    if 'halo' in kernel and 'qfsdp' in kernel:
        wrap_halo_qfsdp(model, config)
    else:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if any([exc in name for exc in exceptions]):
                    print(f"skipped wrapping {name}")
                    continue
                wrapped_module = wrap_linear_module(module, config)
                setattr(wrapped_module, 'layer_name', name)
                swap_module(model, name, wrapped_module)
                print(f"{name} wrapped")

    return orig_model


def unwrap_model(model):
    assert is_wrapped(model), "model is not wrapped"
    delattr(model, "hadamard_config")

    # first unwrap all InputFwdHadamardWrapper modules, if any
    for name, module in model.named_modules():
        if isinstance(module, InputFwdHadamardWrapper):
            swap_module(model, name, module.module)
            print(f"{name} unwrapped")

    # unwrap the rest
    for name, module in model.named_modules():
        if isinstance(module, (SimulatedTwistFormerLinear, JetfireSimLinear, JetfireLinear, SwitchBackLinear, HaloLinear)):
            swap_module(model, name, module.unwrap())
            print(f"{name} unwrapped")

    return model
