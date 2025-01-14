#include <torch/extension.h>


torch::Tensor codebook_quantize(torch::Tensor input, torch::Tensor codebook);

torch::Tensor codebook_quantize_f(torch::Tensor input, torch::Tensor codebook);
