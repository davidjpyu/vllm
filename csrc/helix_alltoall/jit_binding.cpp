/*
 * Pybind11 binding for JIT compilation via torch.utils.cpp_extension.load().
 * This file is NOT used by vLLM's CMake build — it exists solely for
 * standalone testing / benchmarking of the helix all-to-all kernel without
 * building all of vLLM.
 *
 * Usage from Python:
 *   from torch.utils.cpp_extension import load
 *   helix_a2a = load(
 *       name="helix_alltoall",
 *       sources=[
 *           "csrc/helix_alltoall/helix_alltoall.cu",
 *           "csrc/helix_alltoall/helix_alltoall_op.cpp",
 *           "csrc/helix_alltoall/jit_binding.cpp",
 *       ],
 *       extra_include_dirs=["csrc/helix_alltoall"],
 *       extra_cuda_cflags=["-O3", "--use_fast_math",
 *                          "-gencode=arch=compute_90a,code=sm_90a"],
 *       extra_cflags=["-O3"],
 *       verbose=True,
 *   )
 */

#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> alltoall_helix_native(
    torch::Tensor partial_o, torch::Tensor softmax_stats,
    torch::Tensor workspace, int64_t cp_rank, int64_t cp_size);

void initialize_helix_workspace(torch::Tensor workspace, int64_t cp_rank,
                                int64_t cp_size);

int64_t get_helix_workspace_size_per_rank(int64_t cp_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("alltoall_helix_native", &alltoall_helix_native,
        "Native helix all-to-all (two-tensor path)");
  m.def("initialize_helix_workspace", &initialize_helix_workspace,
        "Initialize helix workspace FIFO buffers");
  m.def("get_helix_workspace_size_per_rank",
        &get_helix_workspace_size_per_rank,
        "Workspace size per rank in bytes");
}
