#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <codebook_quant.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <int CODEBOOK_SIZE>
__device__ inline __nv_bfloat16 ternary_search(const __nv_bfloat16* codebook, __nv_bfloat16 val) {
    int left = 0;
    int right = CODEBOOK_SIZE - 1;

    __nv_bfloat16 min_dist = __habs(val - codebook[left]);
    __nv_bfloat16 nearest = codebook[left];

    int mid1;
    int mid2;

    __nv_bfloat16 mid1_val;
    __nv_bfloat16 mid2_val;

    __nv_bfloat16 dist_mid1;
    __nv_bfloat16 dist_mid2;

    while (left <= right) {
        mid1 = left + (right - left) / 3;
        mid2 = right - (right - left) / 3;

        mid1_val = codebook[mid1];
        mid2_val = codebook[mid2];

        dist_mid1 = __habs(val - mid1_val);
        dist_mid2 = __habs(val - mid2_val);

        if (dist_mid1 < min_dist) {
            min_dist = dist_mid1;
            nearest = mid1_val;
        }
        if (dist_mid2 < min_dist) {
            min_dist = dist_mid2;
            nearest = mid2_val;
        }
        if (mid1_val == mid2_val) {
            break;
        }
        if (val < mid1_val) {
            right = mid1 - 1;
        } else if (val > mid2_val) {
            left = mid2 + 1;
        } else {
            left = mid1 + 1;
            right = mid2 - 1;
        }
    }

    for (int i = left; i <= right; ++i) {
        __nv_bfloat16 codebook_val = codebook[i];
        __nv_bfloat16 dist = __habs(val - codebook_val);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = codebook_val;
        }
    }

    return nearest;
}

template <int CODEBOOK_SIZE>
__global__ void codebook_quantize_kernel_no_shared(
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    int num_elements,
    const __nv_bfloat16 *__restrict__ codebook) {

    __nv_bfloat16 s_codebook[CODEBOOK_SIZE];

    #pragma unroll
    for (int i = 0; i < CODEBOOK_SIZE; ++i) {
        s_codebook[i] = codebook[i];
    }

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    const int elements_per_thread = 8;
    int idx = tid * elements_per_thread;

    while (idx < num_elements) {
        unsigned int reg0 = 0, reg1 = 0, reg2 = 0, reg3 = 0;
        int elements_to_process = min(elements_per_thread, num_elements - idx);

        uintptr_t addr = reinterpret_cast<uintptr_t>(&input[idx]);
        bool aligned = (addr % 16) == 0;

        if (elements_to_process == 8 && aligned) {
            asm volatile(
                "ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
                : "l"(input + idx)
            );

            __nv_bfloat16 values[8];
            unsigned int regs[4] = { reg0, reg1, reg2, reg3 };

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned int word = regs[i];
                values[2 * i]     = *reinterpret_cast<__nv_bfloat16*>(&word);
                values[2 * i + 1] = *reinterpret_cast<__nv_bfloat16*>(((char*)&word) + 2);
            }

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                // if CODEBOOK_SIZE is 64, we do ternary_search, otherwise we just use the simple for loop
                if constexpr (CODEBOOK_SIZE > 16) {
                    values[i] = ternary_search<CODEBOOK_SIZE>(s_codebook, values[i]);
                } else {
                    __nv_bfloat16 val = values[i];
                    __nv_bfloat16 min_dist = __habs(val - s_codebook[0]);
                    __nv_bfloat16 nearest = s_codebook[0];

                    #pragma unroll
                    for (int k = 1; k < CODEBOOK_SIZE; ++k) {
                        __nv_bfloat16 dist = __habs(val - s_codebook[k]);
                        if (dist < min_dist) {
                            min_dist = dist;
                            nearest = s_codebook[k];
                        }
                    }
                    values[i] = nearest;
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned int word = 0;
                __nv_bfloat16 v0 = values[2 * i];
                __nv_bfloat16 v1 = values[2 * i + 1];
                memcpy(&word, &v0, 2);
                memcpy(((char*)&word) + 2, &v1, 2);
                regs[i] = word;
            }
            asm volatile(
                "st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
                :
                : "l"(output + idx), "r"(regs[0]), "r"(regs[1]), "r"(regs[2]), "r"(regs[3])
            );
        } else {
            __nv_bfloat16 values[8];
            for (int i = 0; i < elements_to_process; ++i) {
                values[i] = input[idx + i];
            }
            for (int i = 0; i < elements_to_process; ++i) {
                if constexpr (CODEBOOK_SIZE > 16) {
                    values[i] = ternary_search<CODEBOOK_SIZE>(s_codebook, values[i]);
                } else {
                    __nv_bfloat16 val = values[i];
                    __nv_bfloat16 min_dist = __habs(val - s_codebook[0]);
                    __nv_bfloat16 nearest = s_codebook[0];

                    #pragma unroll
                    for (int k = 1; k < CODEBOOK_SIZE; ++k) {
                        __nv_bfloat16 dist = __habs(val - s_codebook[k]);
                        if (dist < min_dist) {
                            min_dist = dist;
                            nearest = s_codebook[k];
                        }
                    }
                    values[i] = nearest;
                }
            }
            for (int i = 0; i < elements_to_process; ++i) {
                output[idx + i] = values[i];
            }
        }
        idx += total_threads * elements_per_thread;
    }
}

torch::Tensor codebook_quantize(torch::Tensor input, torch::Tensor codebook) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(codebook.is_cuda(), "Codebook must be a CUDA tensor");

    TORCH_CHECK(input.dtype() == torch::kBFloat16, "Input tensor must be of type BFloat16.");
    TORCH_CHECK(codebook.dtype() == torch::kBFloat16, "Codebook tensor must be of type BFloat16.");

    const auto num_elements = input.numel();
    const int codebook_size = codebook.numel();

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int elements_per_thread = 8;
    const int total_elements_per_block = threads_per_block * elements_per_thread;
    const int num_blocks = (num_elements + total_elements_per_block - 1) / total_elements_per_block;

    if (codebook_size == 16) {
        codebook_quantize_kernel_no_shared<16><<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            num_elements,
            reinterpret_cast<const __nv_bfloat16*>(codebook.data_ptr<at::BFloat16>())
        );
    } else if (codebook_size == 64) {
        codebook_quantize_kernel_no_shared<64><<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            num_elements,
            reinterpret_cast<const __nv_bfloat16*>(codebook.data_ptr<at::BFloat16>())
        );
    } else {
        TORCH_CHECK(false, "Unsupported codebook size. Supported sizes are 16 and 64.");
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return output;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("codebook_quantize", &codebook_quantize, "Quantize a tensor using a codebook (CUDA)",
//           py::arg("input"), py::arg("codebook"));
// }
