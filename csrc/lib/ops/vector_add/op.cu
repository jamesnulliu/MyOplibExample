#include "../ops.hpp"

namespace my_oplib::cuda
{
template <typename ScalarT>
__global__ void vectorAddKernel(const ScalarT* A, const ScalarT* B, ScalarT* C,
                                std::uint32_t N)
{
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void launchVecAdd(const float* A, const float* B, float* C, std::uint32_t N)
{
    vectorAddKernel<float><<<(N + 255) / 256, 256>>>(A, B, C, N);
}
}  // namespace my_oplib::cuda