from copy import deepcopy

import torch
import qllmt.nn.fn_modules as fn_modules

class SimulatedTwistFormerLinear(torch.nn.Linear):
    def __init__(self, 
                in_features,
                out_features,
                bias=False,
                device=None,
                dtype=None,
                hq_config=None,
                **kwargs):
        super(SimulatedTwistFormerLinear, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        
        self.in_features = in_features
        self.out_features = out_features
        
        assert bias==False, 'Bias is not supported yet'

        self.hq_config = deepcopy(hq_config)
        kernel_name = self.hq_config.get('kernel', 'simulated')
        if kernel_name == 'simulated':
            self._kernel = fn_modules.SimulatedTwistFormerFn
        elif kernel_name == 'fp8_pure':
            self._kernel = fn_modules.LinearPureFP8Fn
        elif kernel_name == 'fp8_hey':
            self._kernel = fn_modules.LinearS4FP8Fn
        elif kernel_name == 'none':
            self._kernel = fn_modules.LinearFn
        else:
            raise ValueError(f'Unsupported kernel: {kernel_name}')

    @staticmethod
    def from_unquantized(module: torch.nn.Linear, hq_config=None, **kwargs):
        q_module = SimulatedTwistFormerLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            hq_config=hq_config,
            **kwargs
        )
        with torch.no_grad():
            q_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                q_module.bias.data.copy_(module.bias.data)
        return q_module


    @torch.no_grad()
    def unwrap(self):
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
        x_view = x.view(-1, x_shape[-1])
        # if hasattr(x, 'quant_scale'):
        #     setattr(x_view, 'quant_scale', x.quant_scale)
        #     setattr(x_view, 'quant_output', x.quant_output.view(-1, x_shape[-1]))
        return self._kernel.apply(
            x_view,
            self.weight,
            self.hq_config,
            getattr(self, 'layer_name', None),
        ).view(*x_shape[:-1], -1)