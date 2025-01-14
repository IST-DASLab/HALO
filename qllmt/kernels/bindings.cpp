#include <torch/extension.h>

// Include all files
#include <gemm.h>
#include <codebook_quant.h>




torch::Tensor linear_a8_w8_bfp32_ofp32(const torch::Tensor &A, 
                                                const torch::Tensor &B,
                                                float alpha)
{
    torch::checkAllContiguous("linear_a8_w8_b32_o32_with_scaling", {{A, "A",       0},
                                                {B, "B", 1}});
    torch::checkDeviceType("linear_a8_w8_b32_o32_with_scaling", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("linear_a8_w8_b32_o32_with_scaling", {{A, "A",       0},
                                          {   B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
    
    return linear_a8_w8_bfp32_ofp32_host(A, B, C, alpha);
}

torch::Tensor linear_a8_w8_b32_o32_with_scaling(const torch::Tensor &A, 
                                                const torch::Tensor &B,
                                                float alpha)
{
    torch::checkAllContiguous("linear_a8_w8_b32_o32_with_scaling", {{A, "A",       0},
                                                {B, "B", 1}});
    torch::checkDeviceType("linear_a8_w8_b32_o32_with_scaling", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("linear_a8_w8_b32_o32_with_scaling", {{A, "A",       0},
                                          {   B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    return linear_a8_w8_b32_o32_with_scaling_host(A, B, C, alpha);
}



/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <utility> // For std::pair

#include "fast_hadamard_transform.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float) {                                    \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

template<typename input_t>
void fast_hadamard_transform_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_quant_transform_cuda(HadamardParamsBaseQuant &params, cudaStream_t stream);


void set_hadamard_params(HadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor out,
                         float scale
                         ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);

    params.scale = scale;
}


void set_hadamard_quant_params(HadamardParamsBaseQuant &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor out,
                         const at::Tensor row_max,
                         float scale
                         ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.row_max = row_max.data_ptr();

    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);

    params.scale = scale;
}


at::Tensor
fast_hadamard_transform(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_cuda<input_t>(params, stream);
    });
    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}





std::pair<at::Tensor, at::Tensor> 
fast_hadamard_transform_int8(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");

    at::Tensor out = torch::empty_like(x);
    // create a tensor of size batch_size
    at::Tensor row_max = torch::zeros({batch_size}, x.device());

    HadamardParamsBaseQuant params;
    set_hadamard_quant_params(params, batch_size, dim, 1, x, out, row_max, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_quant_transform_cuda<input_t>(params, stream);
    });
    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return std::make_pair(out.reshape(shapes_og), row_max);
}


//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{

    m.def("fast_hadamard_transform", &fast_hadamard_transform, "Fast Hadamard transform");
    m.def("fast_hadamard_transform_int8", &fast_hadamard_transform_int8, 
        "Fast Hadamard transform with Int8 Tensor-wise Quantization");

    m.def("linear_a8_w8_b32_o32_with_scaling", &linear_a8_w8_b32_o32_with_scaling,
        "Linear (INT32) with scaling");
    m.def("linear_a8_w8_bfp32_ofp32", &linear_a8_w8_bfp32_ofp32, "Linear (I8-OFP32)");


    // codebook_quant.h
    m.def("codebook_quantize", &codebook_quantize, "Quantize a tensor using a codebook (CUDA)",
          py::arg("input"), py::arg("codebook"));
    m.def("codebook_quantize_f", &codebook_quantize_f, "Quantize a tensor using a codebook (CUDA)",
          py::arg("input"), py::arg("codebook"));
}
