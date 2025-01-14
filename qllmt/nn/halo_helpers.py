import torch
import qllmt
import os

import sys

sys.path.append(os.environ['GEMM_INT8_PATH'])
import gemm_int8

sys.path.append(os.environ['GEMM_FP8_PATH'])
import gemm_fp8

def _precision_to_dtype(precision: str):
    '''
    This function returns the corresponding torch dtype for the given precision (in str).
    '''
    if precision.lower() == 'fp8':
        return torch.float8_e4m3fn
    elif precision.lower() == 'int8':
        return torch.int8
    elif precision.lower() == 'bf16':
        return torch.bfloat16
    elif precision.lower() == 'fp32':
        return torch.float32
    elif precision.lower() == 'fp16':
        return torch.float16
    else:
        raise ValueError(f'Unsupported precision: {precision}')

@torch.compile(dynamic=True)
def contiguous_transpose(x):
    return x.T.contiguous()

def _matmul_fp8_transposed(mat_a, c_a, mat_b, c_b, out_prec=torch.bfloat16):
    """
    Perform matrix multiplication between two 2D matrices in fp8_e4m3fn format.
    """
    assert mat_a.is_contiguous() and mat_b.is_contiguous(), f'Input matrices should be contiguous, mat_a.is_contiguous={mat_a.is_contiguous()}, mat_b.is_contiguous={mat_b.is_contiguous()}'
    assert mat_a.dtype == mat_b.dtype == torch.float8_e4m3fn, f'Only fp8_e4m3fn inputs are supported for now. mat_a.dtype={mat_a.dtype}, mat_b.dtype={mat_b.dtype}'
    if isinstance(c_a, torch.Tensor):
        c_a = c_a.item()
    if isinstance(c_b, torch.Tensor):
        c_b = c_b.item()
    mat_out = gemm_fp8.matmul(mat_a, mat_b, c_a * c_b)
    if out_prec != mat_out.dtype:
        print('WARNING: CASTING THE MATMUL OUTPUT')
        mat_out = mat_out.to(dtype=out_prec)
    return mat_out

def _matmul_int8_transposed(mat_a, c_a, mat_b, c_b, out_prec=torch.bfloat16):
    """
    Perform matrix multiplication between two 2D matrices in int8 format.
    """
    assert mat_a.is_contiguous() and mat_b.is_contiguous(), f'Input matrices should be contiguous, mat_a.is_contiguous={mat_a.is_contiguous()}, mat_b.is_contiguous={mat_b.is_contiguous()}'
    assert mat_a.dtype == mat_b.dtype == torch.int8, f'Only int8 inputs are supported for now. mat_a.dtype={mat_a.dtype}, mat_b.dtype={mat_b.dtype}'
    if isinstance(c_a, torch.Tensor):
        c_a = c_a.item()
    if isinstance(c_b, torch.Tensor):
        c_b = c_b.item()
    mat_out = gemm_int8.matmul(mat_a, mat_b, c_a * c_b)
    if out_prec != mat_out.dtype:
        print('WARNING: CASTING THE MATMUL OUTPUT')
        mat_out = mat_out.to(dtype=out_prec)
    return mat_out

def _matmul_kernel_by_precision(precision):
    if precision == 'fp8':
        return _matmul_fp8_transposed
    elif precision == 'int8':
        return _matmul_int8_transposed
    else:
        raise ValueError(f'Unsupported precision: {precision}')

def quantize_fp8_tranposable(x, transpose=False):
    if transpose:
        return qllmt.functional.quantization.quantize_fp8_tr(x)
    else:
        return qllmt.functional.quantization.quantize_fp8(x)

def quantize_int8_transposable(x, transpose=False):
    if transpose:
        return qllmt.functional.quantization.quantize_int8_tr(x)
    else:
        return qllmt.functional.quantization.quantize_int8(x)

@torch.compile(dynamic=True)
def _fake_quantize_fp8(x):
    return x.to(torch.float8_e4m3fn), x.new_ones(1)

@torch.compile(dynamic=True)
def _fake_quantize_int8(x):
    return x.to(torch.int8), x.new_ones(1)

@torch.compile(dynamic=True)
def _fake_quantize_fp8_tr(x):
    return x.to(torch.float8_e4m3fn).T.contiguous(), x.new_ones(1)

@torch.compile(dynamic=True)
def _fake_quantize_int8_tr(x):
    return x.to(torch.int8).T.contiguous(), x.new_ones(1)

def fake_quantize_fp8_transposable(x, transpose=False):
    if transpose:
        return _fake_quantize_fp8_tr(x)
    else:
        return _fake_quantize_fp8(x)
    
def fake_quantize_int8_transposable(x, transpose=False):
    if transpose:
        return _fake_quantize_int8_tr(x)
    else:
        return _fake_quantize_int8(x)

def _quantize_fn_by_precision(precision, fake_quant=False):
    if fake_quant:
        if precision == 'fp8':
            return fake_quantize_fp8_transposable
        elif precision == 'int8':
            return fake_quantize_int8_transposable
        else:
            raise ValueError(f'Unsupported precision: {precision}')
    else:
        if precision == 'fp8':
            return quantize_fp8_tranposable
        elif precision == 'int8':
            return quantize_int8_transposable
        else:
            raise ValueError(f'Unsupported precision: {precision}')
