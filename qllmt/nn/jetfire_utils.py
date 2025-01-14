import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# The kernel with 1 load operation and 4 store operation
def get_configs_io_block1():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [64, 128]:
            for block_n in [64, 128]:
                num_warps = 4 if block_n <= 64 else 8

                configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n},
                                            num_stages=num_stages, num_warps=num_warps,))
    return configs

@triton.autotune(
    configs=[] + get_configs_io_block1(),
    key=['M', 'N',],
)
@triton.heuristics({
    'BLOCK_SM': lambda args: args["BLOCK_M"] // args["QB"],
    'BLOCK_SN': lambda args: args["BLOCK_N"] // args["QB"],
})
@triton.jit
def int8_quantize_kernel(
                    output_ptr, output_scale_ptr, input_ptr, noise_ptr,
                    M, N, SM, SN,
                    input_stride_0, input_stride_1,
                    output_stride_0, output_stride_1,
                    s_output_stride_0, s_output_stride_1,
                    QB: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_SM: tl.constexpr, BLOCK_SN: tl.constexpr,
                    STOCHASTIC: tl.constexpr,):
    
    # Block PID
    pid = tl.program_id(0)
    NUM_BLOCK_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // NUM_BLOCK_N
    pid_dim1 = pid % NUM_BLOCK_N

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_0, input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    input = tl.load(input_block_ptr).to(tl.float32)
    input = tl.reshape(input, (BLOCK_SM, QB, BLOCK_SN, QB))
    
    # Quantize Scale calculation
    abs_output = tl.abs(input)
    
    # # Fast Max
    max_val = tl.max(abs_output, axis=1) # (1, 1, M, N)
    max_val = tl.max(max_val, axis=2) # （1， 1， M)
    
    # Slow Max
    # max_val = tl.max(abs_output, axis=(1, 3))
    
    scale_output = max_val / 127.
    scale_output = tl.reshape(scale_output, (BLOCK_SM, 1, BLOCK_SN, 1))
    
    # Quantize
    quantize_output = tl.div_rn(input, scale_output)
    quantize_output = tl.reshape(quantize_output, (BLOCK_M, BLOCK_N))

    if STOCHASTIC:
        noise_block_ptr = tl.make_block_ptr(
            base=noise_ptr,
            shape=(M, N),
            strides=(input_stride_0, input_stride_1),
            offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        noise = tl.load(noise_block_ptr)
        quantize_output = _stochastic_rounding(quantize_output, noise)

    quantize_output = libdevice.llrint(quantize_output)
    quantize_output = quantize_output.to(tl.int8)

    scale_output = tl.reshape(scale_output, (BLOCK_SM, BLOCK_SN))
    scale_output = scale_output.to(output_scale_ptr.type.element_ty)

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    scale_output_ptr = tl.make_block_ptr(
        base=output_scale_ptr,
        shape=(SM, SN),
        strides=(s_output_stride_0, s_output_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )

    tl.store(output_block_ptr, quantize_output)
    tl.store(scale_output_ptr, scale_output)

@triton.jit
def _stochastic_rounding(output, noise):
    sign = 1 - 2 * libdevice.signbit(output)
    output = tl.abs(output) + noise

    output = sign * tl.clamp(output, min=-128, max=127)
    
    return output

def int8_quantize(x, QB, stochastic=False):
    if len(x.shape) == 2:
        x_2d = True
    else:
        x_2d = False
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])

    # defining the input and output tensor
    M, N = x.shape
    SM, SN = M // QB, N // QB
    
    y = torch.empty_like(x, dtype=torch.int8, device=x.device)
    s_y = torch.empty((SM, SN), dtype=torch.float16, device=x.device)

    if stochastic:
        noise = torch.empty_like(x, dtype=torch.float32).uniform_(-0.5, 0.5)
    else:
        noise = None

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    int8_quantize_kernel[grid](
        y, s_y, x, noise,
        M, N, SM, SN,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        s_y.stride(0), s_y.stride(1),
        QB, STOCHASTIC=stochastic
    )
    if not x_2d:
        y = y.reshape(BS, -1, y.shape[-1])
        s_y = s_y.reshape(BS, -1, s_y.shape[-1])
    
    return y, s_y





# The kernel with 1 load operation and 4 store operation
def get_configs_io_block2():
    configs = []
    for nstages in [4, 5, 6]:
        for block_m in [64, 128]:
            for block_n in [64, 128]:
                for nwarps in [8, 16, 32]:
                    configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n},
                                                num_stages=nstages, num_warps=nwarps,))
    return configs

@triton.autotune(
    configs=[] + get_configs_io_block2(),
    key=['M', 'N',],
)
@triton.jit
def _int8_transpose_kernel(
                    output_ptr, # output
                    input_ptr, # input
                    M, N, # shape
                    input_stride_0, input_stride_1, # input stride
                    output_stride_0, output_stride_1, # output stride
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,): # CUDA block size
    
    # Block PID
    pid = tl.program_id(0)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // NUM_BLOCK_N
    pid_dim1 = pid % NUM_BLOCK_N

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_0, input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    input = tl.load(input_block_ptr)

    output = tl.trans(input)

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(N, M),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim1 * BLOCK_N, pid_dim0 * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0)
    )

    tl.store(output_block_ptr, output)
  
def int8_transpose(x, transpose_output_2d=False):
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])

    # defining the input and output tensor
    M, N = x.shape
    
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _int8_transpose_kernel[grid](
        y, x,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )

    # Recover 2D to 3D
    if batched and not transpose_output_2d:
        y = y.reshape(BS, -1, y.shape[-1])

    return y





# The kernel with 1 load operation and 4 store operation
def get_configs_io_block3():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        block_m, block_n = 64, 64
        num_warps = 4 if block_n <= 64 else 8

        configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n},
                                      num_stages=num_stages, num_warps=num_warps,))
    return configs

@triton.autotune(
    configs=[] + get_configs_io_block3(),
    key=['M', 'N',],
)
@triton.heuristics({
    'BLOCK_SM': lambda args: args["BLOCK_M"] // args["QB"],
    'BLOCK_SN': lambda args: args["BLOCK_N"] // args["QB"],
})
@triton.jit
def int8_dequantize_kernel(
                    output_ptr, input_ptr, input_scale_ptr,
                    M, N, SM, SN,
                    input_stride_b, input_stride_0, input_stride_1,
                    s_input_stride_b, s_input_stride_0, s_input_stride_1,
                    output_stride_b, output_stride_0, output_stride_1,  
                    QB: tl.constexpr,
                    BLOCK_SM: tl.constexpr, BLOCK_SN: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    
    # Block PID
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)
    NUM_BLOCK_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // NUM_BLOCK_N
    pid_dim1 = pid % NUM_BLOCK_N

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr + pid_b * input_stride_b,
        shape=(M, N),
        strides=(input_stride_0, input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    scale_input_ptr = tl.make_block_ptr(
        base=input_scale_ptr + pid_b * s_input_stride_b,
        shape=(SM, SN),
        strides=(s_input_stride_0, s_input_stride_1),
        offsets=(pid_dim0 * BLOCK_SM, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_SM, BLOCK_SN),
        order=(1, 0),
    )

    input = tl.load(input_block_ptr) # (64, 64)
    input = tl.reshape(input, (BLOCK_SM, QB, BLOCK_SN, QB))

    scale_input = tl.load(scale_input_ptr)
    scale_input = tl.reshape(scale_input, (BLOCK_SM, 1, BLOCK_SN, 1))

    dequantize_output = input * scale_input
    dequantize_output = tl.reshape(dequantize_output, (BLOCK_M, BLOCK_N))
    dequantize_output = dequantize_output.to(tl.float32)

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr + pid_b * output_stride_b,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    tl.store(output_block_ptr, dequantize_output)
  
def int8_dequantize(x, s_x, QB):
    if len(x.shape) == 2:
        x_2d = True
        x = x.unsqueeze(0)
        s_x = s_x.unsqueeze(0)
    else:
        x_2d = False
    # print(x.shape)

    # defining the input and output tensor
    BS, M, N = x.shape
    _, SM, SN = s_x.shape
    
    y = torch.empty_like(x, dtype=torch.float32, device="cuda")

    grid = lambda META: (
        BS, triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    int8_dequantize_kernel[grid](
        y, x, s_x,
        M, N, SM, SN,
        x.stride(0), x.stride(1), x.stride(2),
        s_x.stride(0), s_x.stride(1), s_x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        QB
    )
    if x_2d:
        y = y.squeeze(0)

    return y