#include <cstddef>
#include <cstdint>

namespace my_oplib::cpu
{
void launchVecAdd(const float* A, const float* B, float* C, std::size_t N);

void launchCvtRGBtoGray(std::uint8_t* picOut, const std::uint8_t* picIn,
                        std::uint32_t nRows, std::uint32_t nCols);
}  // namespace my_oplib::cpu

namespace my_oplib::cuda
{
void launchVecAdd(const float* A, const float* B, float* C, std::uint32_t N);

void launchCvtRGBtoGray(std::uint8_t* picOut, const std::uint8_t* picIn,
                        std::uint32_t nRows, std::uint32_t nCols);
}  // namespace my_oplib::cuda