import torch
import torch.nn as nn
from typing import Optional, Any

from ml_collections import ConfigDict

from peft.tuners.lora.layer import LoraLayer
from peft.import_utils import is_qllmt_available
from qllmt.nn.fn_modules import SimulatedTwistFormerFn, QTrainLinearFrozen


# if is_qllmt_available():
# from qllmt.nn import QllmTLinearFrozen


# else:
#     print("QllmTLinearFrozen not available")


# from qllmt.functional.hadamard import matmul_hadU_cuda, get_hadK
# from qllmt.functional.quantization import per_tensor_int8_quant_triton
# import qllmt

def _recursive_dict_update(base_dict, update_dict):
    for k, v in update_dict.items():
        if isinstance(v, dict):
            base_dict[k] = _recursive_dict_update(base_dict.get(k, {}), v)
        else:
            base_dict[k] = v
    return base_dict


class HadamardQuantizedLoraLinear(nn.Module, LoraLayer):
    def __init__(
            self,
            base_layer: nn.Linear,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            hq_config=None,
            **kwargs
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        # base_layer.requires_grad_(False)

        default_hq_config = {
            'fwd': {
                'had_x': 'none',
                'had_w': 'none',
                'quant_x': 'none',
                'quant_w': 'none'
            },
            'bwd1': {
                'had_e': 'none',
                'had_w': 'none',
                'quant_e': 'none',
                'quant_w': 'none'
            },
            'bwd2': {
                'had_e': 'none',
                'had_x': 'none',
                'quant_e': 'none',
                'quant_x': 'none'
            }
        }

        self.hq_config = _recursive_dict_update(default_hq_config, hq_config or {})
        self._kernel = SimulatedTwistFormerFn
        # self._kernel = QTrainLinearFrozen
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora)
        # print(self.hq_config)

    def forward(self, x: torch.Tensor):
        output = self._kernel.apply(
            x.view(-1, x.shape[-1]),
            self.weight,
            self.hq_config
        ).view(*x.shape[:-1], -1)

        # LoRA computation
        lora_A = self.lora_A[self.active_adapter[0]]
        lora_B = self.lora_B[self.active_adapter[0]]
        dropout = self.lora_dropout[self.active_adapter[0]]
        scaling = self.scaling[self.active_adapter[0]]
        x = x.to(lora_A.weight.dtype)
        output = output + lora_B(lora_A(dropout(x))) * scaling

        return output


def hadamard_quantized_lora_linear(
        base_layer: nn.Linear,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        hqt_hq_config: Optional[ConfigDict] = None,
        **kwargs
) -> Optional[HadamardQuantizedLoraLinear]:
    if r > 0:
        return HadamardQuantizedLoraLinear(
            base_layer,
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            hq_config=hqt_hq_config,
            **kwargs
        )
    return None


def dispatch_hqt(
        target: torch.nn.Module,
        adapter_name: str,
        **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, torch.nn.Linear) and kwargs.get('use_hqt', False):
        new_module = hadamard_quantized_lora_linear(target, adapter_name, **kwargs)

    return new_module
