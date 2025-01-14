from functools import partial

import torch
import torch.distributed as dist

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.distributed.fsdp._common_utils import _no_dispatch_record_stream, HandleTrainingState
from torch.distributed.utils import _p_assert

from ..functional.hadamard import right_had
from .halo_helpers import _precision_to_dtype

import warnings
import os, math


def neg_remainder(num, mod):
    if num % mod == 0:
        return 0
    return mod - num % mod

def calculate_fake_tensor_info(layers, n_gpus=None):
    """
    We need to add some fake params to some linear layers to make sure FSDP does not break any row.
    """

    if n_gpus is None:
        # n_gpus = dist.get_world_size()
        n_gpus = int(os.environ['WORLD_SIZE'])
    # print('n_gpus:', n_gpus)
    
    # Calculate total parameters
    remaining_total_params = sum([layer.weight.numel() for layer in layers])

    assert remaining_total_params % n_gpus == 0, f"Total number of parameters {remaining_total_params} is not divisible by number of GPUs {n_gpus}"
    
    params_per_gpu = remaining_total_params // n_gpus
    num_real_params = []
    layers_to_fake_pad_before = [[] for _ in range(n_gpus)]
    layers_to_fake_pad_after = [[] for _ in range(n_gpus)]

    layers_to_fake_pad_before[0].append(0)
    layers_to_fake_pad_after[-1].append(len(layers) - 1)

    overflow_from_last_layer = 0
    overflow_row_size = 1
    next_layer_idx = 0
    running_in_gpu = 0

    for gpu_idx in range(n_gpus):
        if overflow_from_last_layer >= params_per_gpu + neg_remainder(params_per_gpu, overflow_row_size):
            # this gpu can take all its params from the overflow
            pad = neg_remainder(params_per_gpu, overflow_row_size)
            num_real_params.append(params_per_gpu + pad)
            
            overflow_from_last_layer -= num_real_params[-1]
            remaining_total_params -= num_real_params[-1]
            params_per_gpu = remaining_total_params // (n_gpus - gpu_idx) if n_gpus - gpu_idx > 0 else 0
            running_in_gpu = 0
        else:
            # this gpu needs to take some params from the next layer (overflow is not enough)
            if overflow_from_last_layer > 0:
                running_in_gpu = overflow_from_last_layer
                overflow_from_last_layer = 0
                layers_to_fake_pad_after[gpu_idx].append(next_layer_idx - 1)
            
            while True:
                if next_layer_idx >= len(layers):
                    num_real_params.append(running_in_gpu)
                    running_in_gpu = 0
                    break
                
                layer_numel = layers[next_layer_idx].weight.numel()
                layer_row_size = layers[next_layer_idx].weight.shape[1]
                next_layer_idx += 1

                if running_in_gpu + layer_numel < params_per_gpu + neg_remainder(params_per_gpu - running_in_gpu, layer_row_size):
                    # this layer can fully fit in this gpu
                    running_in_gpu += layer_numel
                    layers_to_fake_pad_after[gpu_idx].append(next_layer_idx - 1)
                else:
                    # this layer needs to be split
                    pad = neg_remainder(params_per_gpu - running_in_gpu, layer_row_size)
                    num_real_params.append(params_per_gpu + pad)
                    overflow = running_in_gpu + layer_numel - num_real_params[-1]
                    
                    remaining_total_params -= num_real_params[-1]

                    num_rem_gpus = n_gpus - gpu_idx - 1
                    params_per_gpu = math.ceil(remaining_total_params / num_rem_gpus) if num_rem_gpus > 0 else 0
                    overflow_from_last_layer = overflow
                    overflow_row_size = layer_row_size
                    running_in_gpu = 0

                    break
    
    results = []
    max_gpu_real_params = max(num_real_params)
    for gpu_idx in range(n_gpus):
        num_fake_params = max_gpu_real_params - num_real_params[gpu_idx]
        if num_fake_params == 0:
            continue
        
        if len(layers_to_fake_pad_after[gpu_idx]) > 0:
            results.append({
                'side': 'after',
                'layer_idx': layers_to_fake_pad_after[gpu_idx][0],
                'num_fake_params': num_fake_params
            })
        else:
            assert len(layers_to_fake_pad_before[gpu_idx]) > 0, f"Could not find a layer to fake pad for GPU {gpu_idx}."
            results.append({
                'side': 'before',
                'layer_idx': layers_to_fake_pad_before[gpu_idx][0],
                'num_fake_params': num_fake_params
            })
    
    # print('results:', results)

    return results


@torch.compile(dynamic=True)
def int8_quant_with_scale(x, scale):
    x = x / scale
    x = x.round().to(torch.int8)
    return x


@torch.compile(dynamic=True)
def fused_absmax_scale(x):
    return torch.norm(x, float('inf')) / torch.iinfo(torch.int8).max


@torch.compile(dynamic=True)
def fp8_quant_with_scale(x, scale):
    x = x / scale
    x = x.to(torch.float8_e4m3fn)
    return x


@torch.compile(dynamic=True)
def fused_fp8_absmax_scale(x):
    dtype = torch.float8_e4m3fn
    scale = torch.norm(x, float('inf')) / torch.finfo(dtype).max
    return scale


def calculate_local_scale(x, qdtype):
    if qdtype == torch.int8:
        return fused_absmax_scale(x).item()
    elif qdtype == torch.float8_e4m3fn:
        return fused_fp8_absmax_scale(x)
    else:
        raise ValueError(f'unsupported qdtype: {qdtype}')


def quant_with_scale(x, scale, qdtype):
    if qdtype == torch.int8:
        return int8_quant_with_scale(x, scale)
    elif qdtype == torch.float8_e4m3fn:
        return fp8_quant_with_scale(x, scale)
    else:
        raise ValueError(f'unsupported qdtype: {qdtype}')


def monkey_gather(self, padded_unsharded_flat_param, export_fns, import_fns, qdtype, apply_had, verbose=False):
    """
    All-gather the handle's flat parameter to the destination ``padded_unsharded_flat_param``.

    Then switch to use the all-gathered tensor.
    """

    _p_assert(
        hasattr(self, "process_group") and hasattr(self, "world_size"),
        "Expects a process group and world size to have been set via `shard()`",
    )

    assert qdtype in [torch.int8, torch.float8_e4m3fn], 'only int8 and fp8_e4m3fnuz are supported'

    sharded_flat_param = self.flat_param.data
    expected_numel = sharded_flat_param.numel() * self.world_size
    _p_assert(
        padded_unsharded_flat_param.numel() == expected_numel,
        f"Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}",
    )

    pg = (
        self._fake_process_group
        if self._use_fake_all_gather
        else self.process_group
    )

    # HACK this should be handled by C10D
    in_forward = self._training_state == HandleTrainingState.FORWARD
    assert not sharded_flat_param.is_cpu, "CPU low-precision all-gather is not supported yet"

    for export_fn in export_fns:
        export_fn('in_forward', in_forward)

    verbose = verbose and dist.get_rank() == 0

    if verbose:
        print(f'{dist.get_rank()}: in_forward: {in_forward}')

    device = sharded_flat_param.device
    dtype = sharded_flat_param.dtype

    if verbose:
        print(f'num shards={len(self.flat_param._shard_param_infos)}, len(export_fns)={len(export_fns)}')
    
    local_ws = []
    is_fakes = []
    for shard_info, shape in zip(self.flat_param._shard_param_infos, self.flat_param._shapes):
        # if verbose:
        #     print(f'{dist.get_rank()}: shard info: {shard_info}')

        if not shard_info.in_shard or shard_info.numel_in_shard == 0:
            local_ws.append(None)
            is_fakes.append(len(shape) < 2)
        else:
            start = shard_info.offset_in_shard
            end = start + shard_info.numel_in_shard
            local_w = sharded_flat_param[start:end]

            if len(shape) < 2:
                if verbose:
                    print(f'{dist.get_rank()}: found non-linear layer (possibly a fake padding)')
                is_fakes.append(True)
            else:
                if apply_had:
                    local_w = right_had(local_w.view(-1, shape[-1])).view(-1)
                is_fakes.append(False)
            
            local_ws.append(local_w)
    
    if verbose:
        print(f'{dist.get_rank()}: local_ws: {local_ws}')
        print(f'{dist.get_rank()}: is_fakes: {is_fakes}')

    scale_name = f"{'wH' if apply_had else 'w'}_scale"
    w_scales_precalc = [import_fn(scale_name) for import_fn in import_fns]
    w_needs_scales = not any([s is not None for s in w_scales_precalc])

    if w_needs_scales:
        scales = [calculate_local_scale(local_w, qdtype) if (local_w is not None and not is_fake) else 0 for local_w, is_fake in zip(local_ws, is_fakes)]
        # scales = [1. for _ in range(len(self.flat_param._shard_param_infos))]

        if verbose:
            print(f'{dist.get_rank()}: local scales: {scales}')

        scales = torch.tensor(scales, dtype=dtype, device=device)
        dist.all_reduce(scales, op=torch.distributed.ReduceOp.MAX)

        real_tensor_idx = 0
        for i, is_fake in enumerate(is_fakes):
            if not is_fake:
                export_fns[real_tensor_idx](scale_name, scales[i].item())
                real_tensor_idx += 1
    else:
        if verbose:
            print(f'{dist.get_rank()}: re-using pre-calculated w scales {w_scales_precalc}')

        # make sure to append 0 scales for fake params
        scales = []
        real_tensor_idx = 0
        for is_fake in is_fakes:
            if is_fake:
                scales.append(0)
            else:
                scales.append(w_scales_precalc[real_tensor_idx])
                real_tensor_idx += 1
    
        scales = torch.tensor(scales, dtype=dtype, device=device)

    if verbose:
        print(f'{dist.get_rank()}: global scales: {scales}')

    # dist.all_gather_into_tensor(
    #     padded_unsharded_flat_param,
    #     sharded_flat_param,
    #     pg,
    # )

    assert sharded_flat_param.numel() % 2 == 0, 'only even number of elements per-rank are supported'
    flat_qw = padded_unsharded_flat_param[:sharded_flat_param.numel() // 2].view(qdtype)
    for shard_info, scale, local_w, is_fake in zip(self.flat_param._shard_param_infos, scales, local_ws, is_fakes):
        if not shard_info.in_shard or shard_info.numel_in_shard == 0:
            continue
        if is_fake:
            continue

        start = shard_info.offset_in_shard
        end = start + shard_info.numel_in_shard
        flat_qw[start:end] = quant_with_scale(local_w, scale, qdtype)

    if verbose:
        print(f'{dist.get_rank()}: world size', dist.get_world_size(pg))
        print(f'{dist.get_rank()}: flat_qw size: ', flat_qw.shape)
        print(f'{dist.get_rank()}: sharded_flat_param size: ', sharded_flat_param.shape)
        print(f'{dist.get_rank()}: padded_unsharded_flat_param size: ', padded_unsharded_flat_param.shape)

    qpadded_unsharded_flat_param = torch.empty_like(padded_unsharded_flat_param, dtype=qdtype)
    dist.all_gather_into_tensor(
        qpadded_unsharded_flat_param.view(dtype=torch.int8),
        flat_qw.view(dtype=torch.int8),
        pg,
    )

    padded_unsharded_flat_param.copy_(qpadded_unsharded_flat_param)

    if self._offload_params:
        # In case of offloading, `flat_param.data` (i.e. sharded param) is
        # created on the pre-unshard stream. We need to hand it over to the
        # unshard stream for all-gather
        _no_dispatch_record_stream(
            sharded_flat_param,
            self._device_handle.current_stream(),  # unshard_stream
        )
    return padded_unsharded_flat_param


def patch_fsdp_model(model, qdtype, apply_had=True):
    def _export_fn(name, value, cfg):
        assert cfg is not None, 'Calling export_fn on a module with no hq_config.'
        if 'fsdp_payload' not in cfg:
            cfg['fsdp_payload'] = {}
        cfg['fsdp_payload'][name] = value

    def _import_fn(name, cfg):
        assert cfg is not None, 'Calling import_fn on a module with no hq_config.'
        if 'fsdp_payload' not in cfg:
            return None
        return cfg['fsdp_payload'].get(name, None)

    for name, module in model.named_modules():
        if model != module and isinstance(module, FSDP):
            # find all SimulatedTwistFormerLinear modules in this FSDP block
            # sub_modules = [m for _, m in module.named_modules() if isinstance(m, SimulatedTwistFormerLinear)]
            sub_modules = [m for _, m in module.named_modules() if isinstance(m, torch.nn.Linear)]
            for sub_module in sub_modules:
                assert hasattr(sub_module, 'hq_config'), 'hq_config not found in sub_module. make sure to wrap the module with HaloLinear.from_unquantized'
                assert _precision_to_dtype(sub_module.hq_config['halo_precision']) == qdtype, 'qdtype mismatch for qfsdp'

            if len(sub_modules) == 0:
                print(f'skipping {name} as it has no linear modules')
                continue

            # a very hacky way to pass fsdp-calculated scales to the forward pass
            export_fns = [partial(_export_fn, cfg=getattr(m, 'hq_config', None)) for m in sub_modules]
            import_fns = [partial(_import_fn, cfg=getattr(m, 'hq_config', None)) for m in sub_modules]

            # patch the all_gather function
            patched_gather_fn = monkey_gather.__get__(module._handle, module._handle.__class__)
            setattr(module, '_default_all_gather_flat_param', module._handle._all_gather_flat_param)
            module._handle._all_gather_flat_param = partial(
                patched_gather_fn,
                export_fns=export_fns,
                import_fns=import_fns,
                qdtype=qdtype,
                apply_had=apply_had,
                verbose=False,
            )


def unpatch_fsdp_model(model):
    for name, module in model.named_modules():
        if model != module and isinstance(module, FSDP):
            # unpatch the all_gather function
            if hasattr(module, '_default_all_gather_flat_param'):
                module._handle._all_gather_flat_param = getattr(module, '_default_all_gather_flat_param', None)
                assert module._handle._all_gather_flat_param is not None
                # delattr(module._handle, '_default_all_gather_flat_param')
                print(f'unpatched {name}')

def qfsdp_forward(hq_config, w, with_had=False):
    if 'fsdp_payload' not in hq_config:
        # warnings.warn('FSDP quantized communication is disabled.')
        return None, None
    
    scale_name = f"{'wH' if with_had else 'w'}_scale"
    payload = hq_config['fsdp_payload']

    assert scale_name in payload, f'Expected {scale_name} in fsdp_payload.'
    assert 'halo_precision' in hq_config, 'halo_precision not found in hq_config'

    scale = payload[scale_name]
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=w.device)
    q = w.to(dtype=_precision_to_dtype(hq_config['halo_precision']))

    return q, scale

def qfsdp_backward(hq_config):
    if 'fsdp_payload' in hq_config:
        del hq_config['fsdp_payload']