from copy import deepcopy
import torch
import qllmt
from qllmt.functional.hadamard import right_had
from qllmt.nn.halo_helpers import _matmul_int8_transposed

class LinearInt8(torch.nn.Module):
    '''
        This is for benchmarking purposes only (for now).
        We only support int8 quantization for now.
        
    '''
    def __init__(self, 
                 in_features, out_features,
                 bias=False, dtype=torch.bfloat16,
                 **kwargs):
        '''
        Symmetric 8-bit Linear Layer with (optional) Hadamard on input.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scale',
                             torch.ones((1), requires_grad=False))
        self.register_buffer('weight', (torch.randint(1, 128, (self.out_features, self.in_features),
                                                             dtype=torch.int8, requires_grad=False)))
        self.input_had = kwargs.get('apply_had', False) #it shows that we apply Hadamard to the input or not (default=False)    
        self.input_quant = kwargs.get('apply_quant', True) #it shows that we apply quantization to the input or not (default=True)
        assert bias==False, 'Bias is not supported yet!'
        self.bias = None

    @staticmethod
    def from_unquantized(module: torch.nn.Linear, **kwargs):
        q_module = LinearInt8(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            **kwargs
        )
        # with torch.no_grad():
            # print(f'Copying weights by casting ({module.weight.dtype} -> {torch.int8})!')
            # q_module.weight.data.copy_(module.weight.data.to(torch.int8)) # cast weights to q_dtype
        return q_module


    @torch.no_grad()
    def unwrap(self):
        pass
    
    def forward(self, x):
        
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        
        if self.input_had: #when we need Hadamard, we need quantization as well
            x, scale_x = qllmt.functional.quantization.quantize_int8(right_had(x))
        elif self.input_quant: #we may not need Hadamard but we need quantization
            x, scale_x = qllmt.functional.quantization.quantize_int8(x)
        else: #we may not need Hadamard and quantization
            scale_x = 1.0
            
        y =  _matmul_int8_transposed(
            x, scale_x, 
            self.weight, self.weight_scale,
            )
        y = y.view(*x_shape[:-1], -1)
        return y
