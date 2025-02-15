# ==================================================================================================
# @file compiler-configs-cuda.cmake
# @brief Compiler configurations for cuda.
#
# @note Several parameters SHOULD be set BEFORE including this file:
#         - `ENV{NVCC_CCBIN}`: CUDA Compiler bindir. Default: auto-detected.
#         - `CMAKE_CUDA_STANDARD`: CUDA Standard. Default: 20.
# ==================================================================================================

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES native)
message(STATUS "CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -lineinfo")