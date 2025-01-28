#include <torch/torch.h>

#include "./torch_impl.hpp"

// Define operator `torch.ops.my_torch_op.vector_add`.
// @see
//   https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.fu2gkc7w0nrc
TORCH_LIBRARY(my_oplib, m)
{
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
    m.def("cvt_rgb_to_gray(Tensor img) -> Tensor");
}

// Register the implementation.
// @see
//   https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.jc288bcufw9a
TORCH_LIBRARY_IMPL(my_oplib, CPU, m)
{
    m.impl("vector_add", &my_oplib::cpu::torch_impl::vectorAdd);
    m.impl("cvt_rgb_to_gray", &my_oplib::cpu::torch_impl::cvtRGBtoGray);
}

TORCH_LIBRARY_IMPL(my_oplib, CUDA, m)
{
    m.impl("vector_add", &my_oplib::cuda::torch_impl::vectorAdd);
    m.impl("cvt_rgb_to_gray", &my_oplib::cuda::torch_impl::cvtRGBtoGray);
}