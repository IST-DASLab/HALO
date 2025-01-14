import torch
import qllmt
import random
import math
import qllmt._CUDA as cuda_quant_module


@torch.compile(dynamic=True)
def simulated_global_int8_absmax_quant(x):
    scale = x.abs().max()
    scale = scale.clamp(min=1e-5) / 127
    return (x / scale).clamp(min=-127, max=127).round() * scale


@torch.compile(dynamic=True)
def simulated_col_wise_int8_absmax_quant(x):
    scales = x.abs().max(dim=-2, keepdim=True)[0]
    scales = scales.clamp(min=1e-5) / 127
    return (x / scales).clamp(min=-127, max=127).round() * scales


@torch.compile(dynamic=True)
def simulated_row_wise_int8_absmax_quant(x):
    scales = x.abs().max(dim=-1, keepdim=True)[0]
    scales = scales.clamp(min=1e-5) / 127
    return (x / scales).clamp(min=-127, max=127).round() * scales


@torch.compile(dynamic=True)
def simulated_global_quant_with_scale(x, scale):
    return (x / scale).clamp(min=-127, max=127).round() * scale


MIN_NUMEL = 4096 * 4096




@torch.compile(dynamic=True)
def jetfire_quantize_int8_(x, block_size=32):
    m, n = x.shape
    x = x.reshape(m // block_size, block_size, n // block_size, block_size)
    x = x.permute(0, 2, 1, 3).reshape(-1, block_size * block_size)  # (num_blocks, block_size * block_size)
    scales = x.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5).div(127)  # (num_blocks, 1)
    return (x / scales).clamp(min=-127, max=127).round().to(torch.int8), scales


@torch.compile(dynamic=True)
def jetfire_quantize_fp6_(x, block_size=32):
    m, n = x.shape
    x = x.reshape(m // block_size, block_size, n // block_size, block_size)
    x = x.permute(0, 2, 1, 3).reshape(-1, block_size * block_size)  # (num_blocks, block_size * block_size)
    scales = x.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5).div(28)  # (num_blocks, 1)
    return quantize_fp_codebook((x / scales).clamp(min=-28, max=28), scheme='fp6_e3m2fn', scale=1.), scales

def jetfire_quantize(x, transpose=False, simulate=False, block_size=32, prec='int8'):
    assert prec in ['int8', 'fp6'], f'Unsupported precision: {prec}'
    assert simulate, 'Jetfire quantization is only supported in simulation mode'
    assert not transpose, 'Transpose is not supported for Jetfire quantization'

    org_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    m, n = x.shape

    pad_n = pad_m = 0
    if m % block_size != 0 or n % block_size != 0:
        pad_n = block_size - n % block_size
        pad_m = block_size - m % block_size
        x = torch.nn.functional.pad(x, (0, pad_n, 0, pad_m))
        m, n = x.shape
    assert m % block_size == 0 and n % block_size == 0, f'Block size should be a multiple of the tensor size (m={m}, n={n}, block_size={block_size})'

    q, scales = jetfire_quantize_int8_(x, block_size) if prec == 'int8' else jetfire_quantize_fp6_(x, block_size)

    if simulate:
        q = q.to(x.dtype) * scales
        scales = None
    q = q.reshape(m // block_size, n // block_size, block_size, block_size).permute(0, 2, 1, 3).reshape(m, n)  # (m, n)

    if pad_m > 0 or pad_n > 0:
        q = q[:, :-pad_n] if pad_n > 0 else q
        q = q[:-pad_m] if pad_m > 0 else q

    q = q.reshape(org_shape).contiguous()
    if simulate:
        return q
    else:
        return q, scales


def simulated_quantize_fp8(x, norm_val=1., scale=None, dtype=torch.float8_e4m3fn):
    org_dtype = x.dtype
    if scale is None:
        scale = torch.norm(x, float('inf')) / torch.finfo(dtype).max * norm_val
    x = x / scale
    x = x.to(dtype)
    return x.to(org_dtype) * scale

@torch.compile(dynamic=True)
def quantize_fp8(x, norm_val=1. ):
    dtype=torch.float8_e4m3fn
    scale = torch.norm(x, float('inf')) / torch.finfo(dtype).max * norm_val
    x = x / scale
    x = x.to(dtype)
    return x, scale

@torch.compile(dynamic=True)
def quantize_int8(x, norm_val=1. ):
    dtype=torch.int8
    scale = torch.norm(x, float('inf')) / torch.iinfo(dtype).max * norm_val
    x = x / scale
    x = x.round().to(dtype)
    return x, scale

@torch.compile(dynamic=True)
def quantize_fp8_tr(x, norm_val=1. ):
    dtype=torch.float8_e4m3fn
    scale = torch.norm(x, float('inf')) / torch.finfo(dtype).max * norm_val
    x = x / scale
    x = x.to(dtype)
    return x.T.contiguous(), scale

@torch.compile(dynamic=True)
def quantize_int8_tr(x, norm_val=1. ):
    dtype=torch.int8
    scale = torch.norm(x, float('inf')) / torch.iinfo(dtype).max * norm_val
    x = x / scale
    x = x.round().to(dtype)
    return x.T.contiguous(), scale

def generate_fp_codebook(scheme='fp4_e2m1fn'):
    schemes = {
        'fp4_e3m0fn': (3, 0, 3),
        'fp4_e2m1fn': (2, 1, 1),
        'fp4_e1m2fn': (1, 2, 0),
        'fp6_e3m2fn': (3, 2, 3),
        'fp6_e2m3fn': (2, 3, 1),
    }
    n_e, n_m, bias = schemes.get(scheme, (None, None, None))
    assert n_e is not None, f'Invalid scheme: {scheme}'

    signs = torch.tensor([1.0, -1.0])
    exponents = torch.arange(0, 2 ** n_e)
    mantissas = torch.arange(0, 2 ** n_m)

    codebook = []
    for sign in signs:
        for e in exponents:
            if e == 0:
                E = 1 - bias
                for m in mantissas:
                    if m == 0:
                        value = 0.0
                    else:
                        significand = m / math.pow(2.0, n_m)
                        value = sign * math.pow(2.0, E) * significand
                    codebook.append(value)
            else:
                E = e - bias
                for m in mantissas:
                    significand = 1.0 + m / pow(2.0, n_m)
                    value = sign * math.pow(2.0, E) * significand
                    codebook.append(value)

    codebook = torch.tensor(sorted((codebook)))
    return codebook


def calculate_error(matrix_src, matrix_dst):
    return (torch.norm(matrix_src - matrix_dst) / torch.norm(matrix_src)).item()


# def reconstruct_matrix(mat, scale):
#     mat = mat.clone().to(torch.float32)
#     return mat * scale


fp_codebooks = {
    'fp4_e3m0fn': generate_fp_codebook('fp4_e3m0fn'),
    'fp4_e2m1fn': generate_fp_codebook('fp4_e2m1fn'),
    'fp4_e1m2fn': generate_fp_codebook('fp4_e1m2fn'),
    'fp6_e3m2fn': generate_fp_codebook('fp6_e3m2fn'),
    'fp6_e2m3fn': generate_fp_codebook('fp6_e2m3fn'),
}


def quantize_fp_codebook(x, scheme='fp4_e2m1fn', scale=None, gs_range=10, gs_step=1.1):
    codebook = fp_codebooks.get(scheme).to(device=x.device, dtype=x.dtype)

    if scale is None:
        scale = (torch.norm(x, float('inf')) / torch.norm(codebook, float('inf'))).item() * 0.75
    elif str(scale) == 'gs':
        cur_scale = (torch.norm(x, float('inf')) / torch.norm(codebook, float('inf'))).item()
        best_err = 1e9
        best_scale = cur_scale
        for i in range(gs_range):
            qx = quantize_fp_codebook(x, scheme=scheme, scale=cur_scale)
            error = calculate_error(x, qx)
            if error < best_err:
                best_err = error
                best_scale = cur_scale
            cur_scale /= gs_step
            del qx
        scale = best_scale

    scale = torch.tensor(scale, device=x.device) if not isinstance(scale, torch.Tensor) else scale

    if x.dtype == torch.bfloat16:
        qx = cuda_quant_module.codebook_quantize(x / scale, codebook)
    elif x.dtype == torch.float32:
        qx = cuda_quant_module.codebook_quantize_f(x / scale, codebook)
    else:
        raise ValueError(f'Unsupported CUDA matrix dtype: {x.dtype}')

    return qx * scale


@torch.no_grad()
def quantize_simulate(x, mode, **kwargs):
    if mode == 'none':
        return x
    elif mode in ['int8_absmax_global', 'absmax_global', 'global', 'absmax']:
        return simulated_global_int8_absmax_quant(x)
    elif mode in ['int8_absmax_col_wise', 'absmax_col_wise', 'col_wise']:
        return simulated_col_wise_int8_absmax_quant(x)
    elif mode in ['int8_absmax_row_wise', 'absmax_row_wise', 'row_wise']:
        return simulated_row_wise_int8_absmax_quant(x)
    elif mode in ['int8_block32', 'block32']:
        kwargs = {'block_size': 32, 'simulate': True, **kwargs}
        return jetfire_quantize(x, **kwargs)
    elif mode in ['fp4_e3m0fn', 'fp4_e2m1fn', 'fp4_e1m2fn', 'fp6_e3m2fn', 'fp6_e2m3fn']:
        return quantize_fp_codebook(x, scheme=mode, **kwargs)
    elif mode in ['fp8_e4m3fnuz', 'fp8_e5m2fnuz']:
        dtype_dict = {
            'fp8_e4m3fnuz': torch.float8_e4m3fnuz,
            'fp8_e5m2fnuz': torch.float8_e5m2fnuz,
        }
        return simulated_quantize_fp8(x, dtype=dtype_dict.get(mode), **kwargs)
    else:
        raise ValueError(f'Unknown quantization mode: {mode}')
