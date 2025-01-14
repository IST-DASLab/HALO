import torch
from qllmt.functional.hadamard import left_had, right_had
from .halo_helpers import _matmul_kernel_by_precision, _quantize_fn_by_precision, _precision_to_dtype, contiguous_transpose
from .fsdp import qfsdp_forward, qfsdp_backward


class HaloFnLevel0(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, x, w, hq_config):
        assert (x.dtype == w.dtype == torch.bfloat16), f'Only bfloat16 inputs are supported for now. x.dtype={x.dtype}, w.dtype={w.dtype}'
        precision = hq_config['halo_precision']
        ctx._hq_config = hq_config
        ctx._halo_dtype = _precision_to_dtype(precision)
        ctx._matmul_kernel = _matmul_kernel_by_precision(precision)
        ctx._quantize_fn = _quantize_fn_by_precision(precision, fake_quant=hq_config.get('fake_quant', False))
        
        qw, c_w = qfsdp_forward(ctx._hq_config, w, with_had=False)
        assert qw is None or qw.dtype == ctx._halo_dtype, f'Wrong w dtype received from FSDP.'

        qx, c_x = ctx._quantize_fn(x)
        if qw is None:
            qw, c_w = ctx._quantize_fn(w)

        ctx.save_for_backward(qx, c_x, qw, c_w)
        res = ctx._matmul_kernel(qx, c_x, qw, c_w, out_prec=torch.bfloat16)
        return res

    @staticmethod
    @torch.no_grad()
    def backward(ctx, ey):
        assert ey.dtype == torch.bfloat16, f'Only bfloat16 inputs are supported for now. ey.dtype={ey.dtype}'
        qx, c_x, qw, c_w = ctx.saved_tensors
        ex = gw = None

        qfsdp_backward(ctx._hq_config)
        qey, c_ey = ctx._quantize_fn(ey)

        if ctx.needs_input_grad[0]:
            qwt = contiguous_transpose(qw)
            ex = ctx._matmul_kernel(qey, c_ey, qwt, c_w, out_prec=torch.bfloat16)

        if ctx.needs_input_grad[1]:
            qeyt = contiguous_transpose(qey)
            qxt = contiguous_transpose(qx)
            gw = ctx._matmul_kernel(qeyt, c_ey, qxt, c_x, out_prec=torch.bfloat16)

        return ex, gw, None
