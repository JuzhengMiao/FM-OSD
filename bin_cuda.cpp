#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> bin_cuda_forward(
    torch::Tensor avg_pools,
    int B,
    int sub_desc_dim,
    int num_bins,
    int num_patches0,
    int num_patches1,
    int hierarchy);

// std::vector<torch::Tensor> bin_cuda_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell,
//     torch::Tensor new_cell,
//     torch::Tensor input_gate,
//     torch::Tensor output_gate,
//     torch::Tensor candidate_cell,
//     torch::Tensor X,
//     torch::Tensor gate_weights,
//     torch::Tensor weights);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> bin_forward(
    torch::Tensor avg_pools,
    int B,
    int sub_desc_dim,
    int num_bins,
    int num_patches0,
    int num_patches1,
    int hierarchy) {
  CHECK_INPUT(avg_pools);
  return bin_cuda_forward(avg_pools, B, sub_desc_dim, num_bins, num_patches0, num_patches1, hierarchy);
}

// std::vector<torch::Tensor> lltm_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell,
//     torch::Tensor new_cell,
//     torch::Tensor input_gate,
//     torch::Tensor output_gate,
//     torch::Tensor candidate_cell,
//     torch::Tensor X,
//     torch::Tensor gate_weights,
//     torch::Tensor weights) {
//   CHECK_INPUT(grad_h);
//   CHECK_INPUT(grad_cell);
//   CHECK_INPUT(input_gate);
//   CHECK_INPUT(output_gate);
//   CHECK_INPUT(candidate_cell);
//   CHECK_INPUT(X);
//   CHECK_INPUT(gate_weights);
//   CHECK_INPUT(weights);

//   return lltm_cuda_backward(
//       grad_h,
//       grad_cell,
//       new_cell,
//       input_gate,
//       output_gate,
//       candidate_cell,
//       X,
//       gate_weights,
//       weights);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bin_forward, "bin forward (CUDA)");
//   m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}
