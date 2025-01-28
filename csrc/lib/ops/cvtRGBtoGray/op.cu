#include <cuda_runtime.h>

#include "../ops.hpp"
#include "my_oplib/my_oplib.hpp"

namespace my_oplib::cuda
{

__global__ void cvtRGBtoGrayKernel(std::uint8_t* picOut,
                                   const std::uint8_t* picIn,
                                   std::uint32_t height, std::uint32_t width)
{
    // Suppose each pixel is 3 consecutive chars for the 3 channels (RGB).
    constexpr uint32_t N_CHANNELS = 3;
    // Assign each cuda thread to process one pixel.
    uint32_t row = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row >= height || col >= width) {
        return;
    }

    auto grayOffset = computeOffset<uint32_t>(row, col, height, width);
    uint32_t rgbOffset = grayOffset * N_CHANNELS;

    uint8_t r = picIn[rgbOffset];
    uint8_t g = picIn[rgbOffset + 1];
    uint8_t b = picIn[rgbOffset + 2];

    picOut[grayOffset] = uint8_t(0.21F * r + 0.71F * g + 0.07F * b);
}

void launchCvtRGBtoGray(std::uint8_t* picOut, const std::uint8_t* picIn,
                        std::uint32_t nRows, std::uint32_t nCols)
{
    dim3 blockSize = {32, 32};
    dim3 gridSize = {ceilDiv(nRows, 32), ceilDiv(nCols, 32)};

    cvtRGBtoGrayKernel<<<gridSize, blockSize>>>(picOut, picIn, nRows, nCols);
}

}  // namespace my_oplib::cuda
