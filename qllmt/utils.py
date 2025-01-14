import torch

@torch.compile(dynamic=True)
def contiguous_tr(x):
    return x.transpose(-1, -2).contiguous()


# print('warming up contiguous_tr')
# x = torch.randn(8, 512, 4096).cuda()
# for _ in range(10):
#     contiguous_tr(x)
# del x

class timing_block(torch.profiler.record_function):
    def __init__(self, name, enabled=True, args=None):
        self._enabled = enabled
        self._name = name
        super().__init__('custom-' + name, args)

    def __enter__(self):
        if self._enabled:
            torch.cuda.synchronize()
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._enabled:
            torch.cuda.synchronize()
        return super().__exit__(exc_type, exc_value, traceback)
    

def pack_int8_to_bf16(mat_a, mat_b, mat_out):
    assert mat_a.device == mat_b.device == mat_out.device, "All tensors must be on the same device"
    assert mat_a.shape == mat_b.shape == mat_out.shape, "All tensors must have the same number of elements"

    assert mat_a.dtype == torch.int8, "mat_a must be of dtype torch.int8"
    assert mat_b.dtype == torch.int8, "mat_b must be of dtype torch.int8"
    assert mat_out.dtype == torch.bfloat16, "mat_out must be of dtype torch.bfloat16"

    mat_a_flat = mat_a.flatten()
    mat_b_flat = mat_b.flatten()

    mat_out_int8 = mat_out.view(torch.int8)
    mat_out_int8 = mat_out_int8.view(mat_out_int8.numel())
    mat_out_int8[:mat_a_flat.numel()].copy_(mat_a_flat)
    mat_out_int8[mat_a_flat.numel():].copy_(mat_b_flat)

def unpack_bf16_to_int8(mat_packed):
    assert mat_packed.dtype == torch.bfloat16, "mat_packed must be of dtype torch.bfloat16"

    original_shape = mat_packed.shape
    
    mat_packed_int8 = mat_packed.view(torch.int8).view(2 * mat_packed.numel())

    total_int8_elements = mat_packed_int8.numel()
    half_elements = total_int8_elements // 2

    mat_a_flat = mat_packed_int8[:half_elements]
    mat_b_flat = mat_packed_int8[half_elements:]

    mat_a = mat_a_flat.view(original_shape).to(torch.int8)
    mat_b = mat_b_flat.view(original_shape).to(torch.int8)

    return mat_a, mat_b