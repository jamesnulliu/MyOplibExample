#include "../ops.hpp"
#include "../torch_impl.hpp"

namespace my_oplib::cpu::torch_impl
{
auto vectorAdd(const torch::Tensor& A, const torch::Tensor& B) -> torch::Tensor
{

    torch::Tensor C = torch::zeros_like(A);

    switch (A.scalar_type()) {
    case torch::kFloat32: {
        my_oplib::cpu::launchVecAdd(A.data_ptr<float>(), B.data_ptr<float>(),
                                    C.data_ptr<float>(), A.flatten().size(0));
        break;
    }
    default:
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }

    return C;
}
}  // namespace my_oplib::cpu::torch_impl

namespace my_oplib::cuda::torch_impl
{
auto vectorAdd(const torch::Tensor& A, const torch::Tensor& B) -> torch::Tensor
{

    torch::Tensor C = torch::empty_like(A);

    switch (A.scalar_type()) {
    case torch::kFloat32: {
        my_oplib::cuda::launchVecAdd(A.data_ptr<float>(), B.data_ptr<float>(),
                                     C.data_ptr<float>(), A.flatten().size(0));
        break;
    }
    default:
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }

    return C;
}
}  // namespace my_oplib::cuda::torch_impl