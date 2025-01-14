import torch
import torch.nn.functional as F
from qllmt.functional.quantization import jetfire_quantize

class JetfireSimFunction(torch.autograd.Function):
    '''
    Class for JetfireSimFunction
    '''
    @staticmethod
    def forward(ctx, x, w, prec):
        qxH = jetfire_quantize(x, simulate=True, prec=prec)
        qwH = jetfire_quantize(w, simulate=True, prec=prec)
        
        ctx.save_for_backward(qxH, qwH)
        ctx.prec = prec

        out = F.linear(qxH, qwH)
        out = jetfire_quantize(out, simulate=True, prec=prec)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        qxH, qwH = ctx.saved_tensors
        prec = ctx.prec

        grad_input = grad_weight = None
        qey = jetfire_quantize(grad_output, simulate=True, prec=prec)

        if ctx.needs_input_grad[0]:
            grad_input = F.linear(qey, qwH.T)
            grad_input = jetfire_quantize(grad_input, simulate=True, prec=prec)
            
        if ctx.needs_input_grad[1]:
            grad_weight = F.linear(qey.T, qxH.T)

        return grad_input, grad_weight, None
    

class JetfireSimLinear(torch.nn.Linear):
    def __init__(self, 
                in_features,
                out_features,
                bias=False,
                device=None,
                precision=None):
        super(JetfireSimLinear, self).__init__(in_features, out_features, bias, device)
        
        assert precision in ['int8', 'fp6'], f'Unsupported precision {precision}'
        assert not bias, "JetfireSimLinear does not support bias yet"
        self.in_features = in_features
        self.out_features = out_features
        self.prec = precision

    @staticmethod
    def from_unquantized(module: torch.nn.Linear, precision: str):
        q_module = JetfireSimLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            precision=precision
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
        x_shape = x.shape
        return JetfireSimFunction.apply(
            x.view(-1, x_shape[-1]),
            self.weight,
            self.prec
        ).view(*x_shape[:-1], -1)