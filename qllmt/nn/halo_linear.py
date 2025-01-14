from copy import deepcopy

import torch
from qllmt.nn import halo0_fns, halo1_fns, halo2_fns

class HaloLinear(torch.nn.Linear):
    def __init__(self, 
                in_features,
                out_features,
                bias=False,
                device=None,
                dtype=None,
                hq_config=None,
                **kwargs):
        
        super(HaloLinear, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        if hq_config.get('fake_before', 0) > 0:
            raise NotImplementedError('fake_before is not supported yet')
        #     self.fake_before = torch.nn.Parameter(torch.zeros(hq_config['fake_before'], dtype=dtype, device=device))

        if hq_config.get('fake_after', 0) > 0:
            self.fake_after = torch.nn.Parameter(torch.zeros(hq_config['fake_after'], dtype=dtype, device=device))
        
        self.in_features = in_features
        self.out_features = out_features
        
        assert bias==False, 'Bias is not supported yet'

        self.hq_config = deepcopy(hq_config)
        kernel_name = self.hq_config.get('kernel', 'halo0_fp8')
        
        assert 'fp8' in kernel_name or 'int8' in kernel_name, f'Unsupported precision: {kernel_name}'
        self.hq_config['halo_precision'] = 'fp8' if 'fp8' in kernel_name else 'int8'
        backward_xH = True if 'bxh' in kernel_name.lower() else False

        if kernel_name.startswith('halo0') or kernel_name.startswith('haloi'):
            self.hq_config['fake_quant'] = kernel_name.startswith('haloi')
            self._fn = halo0_fns.HaloFnLevel0
        elif kernel_name.startswith('halo1'):
            # self._fn = halo1_fns.HaloFnLevel1WithQFSDPBackwardXH if backward_xH else halo1_fns.HaloFnLevel1WithQFSDP
            self._fn = halo1_fns.HaloFnLevel1WithQFSDPBackwardXH
        elif kernel_name.startswith('halo2'):
            # self._fn = halo2_fns.HaloFnLevel2WithQFSDPBackwardXH if backward_xH else halo2_fns.HaloFnLevel2
            self._fn = halo2_fns.HaloFnLevel2WithQFSDPBackwardXH
        else:
            raise ValueError(f'Unsupported Halo level: {kernel_name}')

    @staticmethod
    def from_unquantized(module: torch.nn.Linear, hq_config=None, **kwargs):
        q_module = HaloLinear(
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
        return self._fn.apply(
            x_view,
            self.weight,
            self.hq_config,
        ).view(*x_shape[:-1], -1)