import torch
import BlockQuantizeCUDA as BQC
from .jetfire_utils import int8_quantize, int8_transpose, int8_dequantize

def int8_linear_forward(X8_3D, SX16_3D, W8, SW16):
    X8_3D_shape = X8_3D.shape
    SX16_3D_shape = SX16_3D.shape
    O_3D_shape = list(X8_3D_shape)
    SO16_3D_shape = list(SX16_3D_shape)
    O_3D_shape[-1] = W8.shape[0]

    C_in = X8_3D.shape[-1]
    
    X8 = X8_3D.reshape(-1, C_in)
    SO16_3D_shape[-1] = SW16.shape[0]

    SX16 = SX16_3D.reshape(-1, SX16_3D.shape[-1])
    SX16T = SX16.t().contiguous()
    SW16T = SW16.t().contiguous()

    bias = torch.zeros((W8.shape[0]), device=W8.device)
    biasmax = torch.zeros((1,), device=W8.device)

    O8, SO16 = BQC.igemm_output_int_quantize_bias_rowcol(X8, W8.t(), bias, biasmax, SX16T, SW16T, X8.shape[0], W8.shape[0], X8.shape[1])

    O8_3D = O8.reshape(O_3D_shape)
    SO16_3D = SO16.reshape(SO16_3D_shape)
    return O8_3D, SO16_3D

def int8_linear_backward(X8_3D, SX16_3D, G8_3D, SG16_3D, G8T, W8, SW16):
    GX8_3D_shape = list(G8_3D.shape)
    GX8_3D_shape[-1] = W8.shape[1]
    GSX16_3D_shape = list(SG16_3D.shape)
    GSX16_3D_shape[-1] = SW16.shape[1]

    SG16 = SG16_3D.reshape(-1, SG16_3D.shape[-1])
    SG16T = SG16.t().contiguous()
    
    X8 = X8_3D.reshape(-1, X8_3D.shape[-1])
    SX16 = SX16_3D.reshape(-1, SX16_3D.shape[-1])
    G8 = G8_3D.reshape(-1, G8_3D.shape[-1])

    GX8, GSX16 = BQC.igemm_output_int_quantize(G8, W8, SG16T, SW16, G8.shape[0], W8.shape[1], G8.shape[1])

    GW = BQC.igemm_output_fp_no_quantize(G8T, X8, SG16, SX16, G8T.shape[0], X8.shape[1], G8T.shape[1])

    GX8_3D = GX8.reshape(GX8_3D_shape)
    GSX16_3D = GSX16.reshape(GSX16_3D_shape)

    return GX8_3D, GSX16_3D, GW

class JetfireFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, qx, sx, w, block_size):
        qw, sw = int8_quantize(w, block_size)
        qx = qx.view(torch.int8)
        ctx.save_for_backward(qx, sx, qw, sw)
        ctx.block_size = block_size

        qy, sy = int8_linear_forward(qx, sx, qw, sw)
        return qy.view(torch.float8_e4m3fn), sy

    @staticmethod
    def backward(ctx, qg, sg):
        qx, sx, qw, sw = ctx.saved_tensors
        block_size = ctx.block_size
        qg = qg.view(torch.int8)

        qgt = int8_transpose(qg, transpose_output_2d=True)
        qgx, sgx, gw = int8_linear_backward(qx, sx, qg, sg, qgt, qw, sw)
        qgx = qgx.view(torch.float8_e4m3fn)
        return qgx, sgx, gw, None
    

class JetfireLinear(torch.nn.Linear):
    def __init__(self, 
                in_features,
                out_features,
                bias=False,
                device=None):
        super(JetfireLinear, self).__init__(in_features, out_features, bias, device)
        
        assert not bias, "JetfireLinear does not support bias yet"
        self.in_features = in_features
        self.out_features = out_features

    @staticmethod
    def from_unquantized(module: torch.nn.Linear):
        q_module = JetfireLinear(
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

    
    def forward(self, qx, sx):
        block_size = 32
        qy, sy = JetfireFn.apply(qx, sx, self.weight, block_size)
        return qy, sy