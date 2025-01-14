import torch
from qllmt.functional.hadamard import left_had, right_had
from .halo_helpers import _matmul_kernel_by_precision, _quantize_fn_by_precision, _precision_to_dtype, contiguous_transpose
from .fsdp import qfsdp_forward, qfsdp_backward

class HaloFnLevel2(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, x, w, hq_config):
        assert (x.dtype == w.dtype == torch.bfloat16), f'Only bfloat16 inputs are supported for now. x.dtype={x.dtype}, w.dtype={w.dtype}'
        precision = hq_config['halo_precision']
        ctx._hq_config = hq_config
        ctx._halo_dtype = _precision_to_dtype(precision)
        ctx._matmul_kernel = _matmul_kernel_by_precision(precision)
        ctx._quantize_fn = _quantize_fn_by_precision(precision)
        
        wH = right_had(w)
        qwH, c_wH = ctx._quantize_fn(wH)
            
        xH = right_had(x)
        qxH, c_xH = ctx._quantize_fn(xH)

        ctx.save_for_backward(x, w)
        res = ctx._matmul_kernel(qxH, c_xH, qwH, c_wH, out_prec=torch.bfloat16)
        return res

    @staticmethod
    @torch.no_grad()
    def backward(ctx, ey):
        assert ey.dtype == torch.bfloat16, f'Only bfloat16 inputs are supported for now. ey.dtype={ey.dtype}'
        x, w = ctx.saved_tensors
        ex = gw = None

        if ctx.needs_input_grad[0]:
            Hey_t = right_had(contiguous_transpose(ey))
            qHey, c_Hey = ctx._quantize_fn(Hey_t, transpose=True)
            qw_t, c_w = ctx._quantize_fn(w, transpose=True)
            Hex_t = ctx._matmul_kernel(qw_t, c_w, qHey, c_Hey, out_prec=torch.bfloat16)
            ex_t = right_had(Hex_t, transpose=True)
            ex = contiguous_transpose(ex_t)

        if ctx.needs_input_grad[1]:
            qey_t, c_ey = ctx._quantize_fn(ey, transpose=True)
            qx_t, c_x = ctx._quantize_fn(x, transpose=True)
            gw = ctx._matmul_kernel(qey_t, c_ey, qx_t, c_x, out_prec=torch.bfloat16)

        return ex, gw, None
    

class HaloFnLevel2WithQFSDP(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, x, w, hq_config):
        assert (x.dtype == w.dtype == torch.bfloat16), f'Only bfloat16 inputs are supported for now. x.dtype={x.dtype}, w.dtype={w.dtype}'
        precision = hq_config['halo_precision']
        ctx._hq_config = hq_config
        ctx._halo_dtype = _precision_to_dtype(precision)
        ctx._matmul_kernel = _matmul_kernel_by_precision(precision)
        ctx._quantize_fn = _quantize_fn_by_precision(precision)
        
        qwH, c_wH = qfsdp_forward(ctx._hq_config, w, with_had=True)
        assert qwH is None or qwH.dtype == ctx._halo_dtype, f'Wrong w dtype received from FSDP.'
        if qwH is None:
            wH = right_had(w)
            qwH, c_wH = ctx._quantize_fn(wH)
            
        xH = right_had(x)
        qxH, c_xH = ctx._quantize_fn(xH)

        ctx.save_for_backward(x, qwH, c_wH)
        res = ctx._matmul_kernel(qxH, c_xH, qwH, c_wH, out_prec=torch.bfloat16)
        return res

    @staticmethod
    @torch.no_grad()
    def backward(ctx, ey):
        assert ey.dtype == torch.bfloat16, f'Only bfloat16 inputs are supported for now. ey.dtype={ey.dtype}'
        x, qwH, c_wH = ctx.saved_tensors
        ex = gw = None

        qfsdp_backward(ctx._hq_config)

        if ctx.needs_input_grad[0]:
            Hey_t = right_had(contiguous_transpose(ey))
            qHey, c_Hey = ctx._quantize_fn(Hey_t, transpose=True)
            qwH_t = contiguous_transpose(qwH)
            HexH_t = ctx._matmul_kernel(qwH_t, c_wH, qHey, c_Hey, out_prec=torch.bfloat16)
            exH_t = right_had(HexH_t, transpose=True)
            exH = contiguous_transpose(exH_t)
            ex = right_had(exH, transpose=True)

        if ctx.needs_input_grad[1]:
            qey_t, c_ey = ctx._quantize_fn(ey, transpose=True)
            qx_t, c_x = ctx._quantize_fn(x, transpose=True)
            gw = ctx._matmul_kernel(qey_t, c_ey, qx_t, c_x, out_prec=torch.bfloat16)

        return ex, gw, None

class HaloFnLevel2WithQFSDPBackwardXH(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, x, w, hq_config):
        assert (x.dtype == w.dtype == torch.bfloat16), f'Only bfloat16 inputs are supported for now. x.dtype={x.dtype}, w.dtype={w.dtype}'
        precision = hq_config['halo_precision']
        ctx._hq_config = hq_config
        ctx._halo_dtype = _precision_to_dtype(precision)
        ctx._matmul_kernel = _matmul_kernel_by_precision(precision)
        ctx._quantize_fn = _quantize_fn_by_precision(precision)
        
        qwH, c_wH = qfsdp_forward(ctx._hq_config, w, with_had=True)
        assert qwH is None or qwH.dtype == ctx._halo_dtype, f'Wrong w dtype received from FSDP.'
        if qwH is None:
            wH = right_had(w)
            qwH, c_wH = ctx._quantize_fn(wH)
            
        xH = right_had(x)
        qxH, c_xH = ctx._quantize_fn(xH)

        ctx.save_for_backward(qxH, c_xH, qwH, c_wH)
        res = ctx._matmul_kernel(qxH, c_xH, qwH, c_wH, out_prec=torch.bfloat16)
        return res

    @staticmethod
    @torch.no_grad()
    def backward(ctx, ey):
        assert ey.dtype == torch.bfloat16, f'Only bfloat16 inputs are supported for now. ey.dtype={ey.dtype}'
        qxH, c_xH, qwH, c_wH = ctx.saved_tensors
        ex = gw = None

        qfsdp_backward(ctx._hq_config)

        if ctx.needs_input_grad[0]:
            Hey_t = right_had(contiguous_transpose(ey))
            qHey, c_Hey = ctx._quantize_fn(Hey_t, transpose=True)
            qwH_t = contiguous_transpose(qwH)
            HexH_t = ctx._matmul_kernel(qwH_t, c_wH, qHey, c_Hey, out_prec=torch.bfloat16)
            exH_t = right_had(HexH_t, transpose=True)
            exH = contiguous_transpose(exH_t)
            ex = right_had(exH, transpose=True)

        if ctx.needs_input_grad[1]:
            qey_t, c_ey = ctx._quantize_fn(ey, transpose=True)
            qxH_t = contiguous_transpose(qxH)
            gwH = ctx._matmul_kernel(qey_t, c_ey, qxH_t, c_xH, out_prec=torch.bfloat16)
            gw = right_had(gwH, transpose=True)

        return ex, gw, None
