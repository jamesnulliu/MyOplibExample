# ==================================================================================================
# @brief Enable CUDA support.
# @see "https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html"
# 
# @note Several parameters should be set before including this file:
#       - ENV{CUDA_HOME}:
#             Path to spdlog libaray installation path.
# ==================================================================================================

# Append the path to the CMAKE_PREFIX_PATH
set(CUDA_CMAKE_PREFIX_PATH "$ENV{CUDA_HOME}/lib64/cmake")
list(APPEND CMAKE_PREFIX_PATH ${CUDA_CMAKE_PREFIX_PATH})

# Find the CUDA package
# @see 
# |- All imported targets: 
# |- "https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html#imported-targets"
find_package(CUDAToolkit REQUIRED)