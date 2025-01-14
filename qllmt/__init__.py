import torch
from . import nn
from . import functional


import qllmt._CUDA


__all__ = [ 
           "linear_a8_w8_b32_o32_with_scaling", 
           "linear_a8_w8_bfp32_ofp32",
           "power_two_had",
           "power_two_fwd_had"
]


class HadamardTransformFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale=1.0):
        ctx._hadamard_transform_scale = scale
        return qllmt._CUDA.fast_hadamard_transform(x, scale)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return qllmt._CUDA.fast_hadamard_transform(dout, ctx._hadamard_transform_scale), None

class FwdHadamardTransformFn(torch.autograd.Function):
    '''
        Hadamard only forward pass
    '''

    @staticmethod
    def forward(ctx, x, scale=1.0):
        return qllmt._CUDA.fast_hadamard_transform(x, scale)

    @staticmethod
    def backward(ctx, dout):
        return dout, None



def linear_a8_w8_b32_o32_with_scaling(input: torch.Tensor,
                                        weight: torch.Tensor,
                                        alpha):
    return qllmt._CUDA.linear_a8_w8_b32_o32_with_scaling(input, weight, alpha)

def linear_a8_w8_bfp32_ofp32(input: torch.Tensor,
                                weight: torch.Tensor,
                                alpha):
    
    
    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    return qllmt._CUDA.linear_a8_w8_bfp32_ofp32(input, weight, alpha)

def power_two_had(input: torch.Tensor, 
                  scale: float=1.0):
    return HadamardTransformFn.apply(input, scale)
    
def power_two_fwd_had(input: torch.Tensor, 
                  scale: float=1.0):
    return FwdHadamardTransformFn.apply(input, scale)

@torch.compile(dynamic=True)
def power_two_had_int8(input: torch.Tensor, 
                  scale: float=1.0):
    # assert functional.hadamard.is_pow2(input.shape[-1]), "Input shape must be a power of 2 for Quant!!"
    y_had, row_scales = qllmt._CUDA.fast_hadamard_transform_int8(input, scale)
    global_scale = row_scales.max()
    return y_had.div_(global_scale).round_().to(torch.int8), global_scale
    