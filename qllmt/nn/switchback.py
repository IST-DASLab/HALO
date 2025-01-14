import torch
import math
import triton
import triton.language as tl

from bitsandbytes.triton.quantize_rowwise import _quantize_rowwise
from bitsandbytes.triton.quantize_global import quantize_global, quantize_global_transpose
from bitsandbytes.triton.int8_matmul_mixed_dequantize import _int8_matmul_mixed_dequantize

def quantize_rowwise_bf16(x: torch.Tensor):
        output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
        output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.bfloat16)

        P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

        assert x.is_cuda and output.is_cuda
        n_elements = output.numel()
        grid = lambda meta: (x.shape[0],)
        _quantize_rowwise[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
        return output, output_maxs

def int8_matmul_mixed_dequantize_bf16(a, b, state_x, state_w, bias):
        device = a.device
        divfactor = 1.0 / (127.0 * 127.0)
        has_bias = 0 if bias is None else 1
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        c = torch.empty((M, N), device=device, dtype=torch.bfloat16)
        # accumulator types
        ACC_TYPE = tl.float32  # if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # launch int8_matmul_mixed_dequantize kernel
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), META["SPLIT_K"])
        _int8_matmul_mixed_dequantize[grid](
            a,
            b,
            c,
            bias,
            state_x,
            state_w,
            M,
            N,
            K,
            divfactor,
            has_bias,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            GROUP_M=8,
            ACC_TYPE=ACC_TYPE,
        )
        return c

if hasattr(tl.math, 'llrint'):
    setattr(tl, 'libdevice', tl.math)
else:
    setattr(tl, 'libdevice', tl.extra.cuda.libdevice)


class _switchback_global_bf16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_3D, W, bias):
        # reshape input to [N * L, D]
        X = X_3D.view(-1, X_3D.size(-1))

        # rowwise quantize for X, global quantize for W
        X_int8, state_X = quantize_rowwise_bf16(X)
        W_int8, state_W = quantize_global(W)

        # save for backward.
        ctx.save_for_backward = X, W

        # matmult, fused dequant and add bias
        # call "mixed" because we are mixing rowwise quantized and global quantized
        return int8_matmul_mixed_dequantize_bf16(X_int8, W_int8.t(), state_X, state_W, bias).view(*X_3D.size()[:-1], -1)

    @staticmethod
    def backward(ctx, G_3D):
        # reshape input to [N_out * L, D]
        G = G_3D.reshape(-1, G_3D.size(-1))

        grad_X = grad_W = grad_bias = None

        X, W = ctx.save_for_backward
        if ctx.needs_input_grad[0]:
            # rowwise quantize for G, global quantize for W
            # for W, we also fuse the transpose operation because only A @ B^T is supported
            # so we transpose once then call .t() in the matmul
            G_int8, state_G = quantize_rowwise_bf16(G)
            W_int8, state_W = quantize_global_transpose(W)
            grad_X = int8_matmul_mixed_dequantize_bf16(G_int8, W_int8.t(), state_G, state_W, None).view(
                *G_3D.size()[:-1],
                -1,
            )
        if ctx.needs_input_grad[1]:
            # backward pass uses standard weight grad
            grad_W = torch.matmul(G.t(), X.to(G.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias


class SwitchBackLinear(torch.nn.Linear):
    def __init__(self, 
                in_features,
                out_features,
                bias=False,
                device=None):
        super(SwitchBackLinear, self).__init__(in_features, out_features, bias, device)
        
        self.in_features = in_features
        self.out_features = out_features

    @staticmethod
    def from_unquantized(module: torch.nn.Linear):
        q_module = SwitchBackLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None
        ).to(module.weight.dtype).to(module.weight.device)
        q_module.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            q_module.bias.data.copy_(module.bias.data)
        return q_module


    def unwrap(self):
        with torch.no_grad():
            bias = getattr(self, 'bias', None)
            module = torch.nn.Linear(
                self.in_features, 
                self.out_features,
                bias=bias is not None,
                device=self.weight.device,
                dtype=self.weight.dtype
            )
            module.weight.data.copy_(self.weight.data)
            if bias is not None:
                module.bias.data.copy_(bias.data)
            return module

    
    def forward(self, x):
        return _switchback_global_bf16.apply(x, self.weight, self.bias)