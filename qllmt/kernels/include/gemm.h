#pragma once

#include <common.h>
#include <torch/types.h>


// used by out_proj and fc2, return INT32
torch::Tensor linear_a8_w8_b32_o32_with_scaling_host(torch::Tensor input,  // INT8
                                                torch::Tensor weight, // INT8
                                                torch::Tensor out,   // INT32
                                                float alpha          // FP32
);
// used by out_proj and fc2, return FP32
torch::Tensor linear_a8_w8_bfp32_ofp32_host(torch::Tensor input,  // INT8
                                                torch::Tensor weight, // INT8
                                                torch::Tensor out,   // INT32
                                                float alpha          // FP32
);
