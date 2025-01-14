#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
__global__ void codebook_quantize_kernel_no_shared_f(
    const float *__restrict__ input,
    float *__restrict__ output,
    int num_elements,
    const float *__restrict__ codebook) {

    // Load codebook into per-thread registers
    float s_codebook[CODEBOOK_SIZE];

    #pragma unroll
    for (int i = 0; i < CODEBOOK_SIZE; ++i) {
        s_codebook[i] = codebook[i];
    }

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    int elements_per_thread = 4;  // Each thread processes 4 elements
    int idx = tid * elements_per_thread;

    // Process elements without using shared memory
    while (idx < num_elements) {
        float values[4];
        int elements_to_process = min(elements_per_thread, num_elements - idx);

        // Load data from global memory
        for (int i = 0; i < elements_to_process; ++i) {
            values[i] = input[idx + i];
        }

        // Quantize each element
        for (int i = 0; i < elements_to_process; ++i) {
            float val = values[i];
            float min_dist = fabsf(val - s_codebook[0]);
            float nearest = s_codebook[0];

            #pragma unroll
            for (int k = 1; k < CODEBOOK_SIZE; ++k) {
                float dist = fabsf(val - s_codebook[k]);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest = s_codebook[k];
                }
            }
            values[i] = nearest;
        }

        // Write results back to global memory
        for (int i = 0; i < elements_to_process; ++i) {
            output[idx + i] = values[i];
        }

        // Move to the next set of elements
        idx += total_threads * elements_per_thread;
    }
}

torch::Tensor codebook_quantize_f(torch::Tensor input, torch::Tensor codebook) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(codebook.is_cuda(), "Codebook must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be of type Float32.");
    TORCH_CHECK(codebook.dtype() == torch::kFloat32, "Codebook tensor must be of type Float32.");

    const auto num_elements = input.numel();
    const int codebook_size = codebook.numel();

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int elements_per_thread = 4;
    const int total_elements_per_block = threads_per_block * elements_per_thread;
    const int num_blocks = (num_elements + total_elements_per_block - 1) / total_elements_per_block;

    if (codebook_size == 16) {
        codebook_quantize_kernel_no_shared_f<16><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            num_elements,
            codebook.data_ptr<float>()
        );
    } else if (codebook_size == 64) {
        codebook_quantize_kernel_no_shared_f<8><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            num_elements,
            codebook.data_ptr<float>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported codebook size. Supported sizes are 16 and 64.");
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return output;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("codebook_quantize_f", &codebook_quantize_f, "Quantize a tensor using a codebook (CUDA)",
//           py::arg("input"), py::arg("codebook"));
// }
