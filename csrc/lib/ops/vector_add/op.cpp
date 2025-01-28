#include "../ops.hpp"

namespace my_oplib::cpu
{
void launchVecAdd(const float* A, const float* B, float* C, std::size_t N)
{
#pragma omp parallel for
    for (std::size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}
}  // namespace my_oplib::cpu