#include <torch/torch.h>

namespace my_oplib::cpu::torch_impl
{
auto vectorAdd(const torch::Tensor& A,
               const torch::Tensor& B) -> torch::Tensor;

auto cvtRGBtoGray(const torch::Tensor& img) -> torch::Tensor;
}  // namespace my_oplib::cpu::torch_impl

namespace my_oplib::cuda::torch_impl
{
auto vectorAdd(const torch::Tensor& A,
               const torch::Tensor& B) -> torch::Tensor;

auto cvtRGBtoGray(const torch::Tensor& img) -> torch::Tensor;
}  // namespace my_oplib::cuda::torch_impl