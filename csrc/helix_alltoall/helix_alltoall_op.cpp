/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include "helix_alltoall.h"

std::tuple<torch::Tensor, torch::Tensor> alltoall_helix_native(
    torch::Tensor partial_o, torch::Tensor softmax_stats,
    torch::Tensor workspace, int64_t cp_rank, int64_t cp_size) {
  TORCH_CHECK(partial_o.is_cuda(), "partial_o must be on CUDA");
  TORCH_CHECK(softmax_stats.is_cuda(), "softmax_stats must be on CUDA");
  TORCH_CHECK(workspace.is_cuda(), "workspace must be on CUDA");
  TORCH_CHECK(partial_o.is_contiguous(), "partial_o must be contiguous");
  TORCH_CHECK(softmax_stats.is_contiguous(),
              "softmax_stats must be contiguous");

  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Half ||
                  partial_o.scalar_type() == at::ScalarType::BFloat16,
              "partial_o must be half or bfloat16");
  TORCH_CHECK(softmax_stats.scalar_type() == at::ScalarType::Float,
              "softmax_stats must be float32");
  TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Long,
              "workspace must be int64 (used as uint64)");

  TORCH_CHECK(partial_o.dim() >= 2,
              "partial_o must have at least 2 dimensions");
  TORCH_CHECK(softmax_stats.dim() >= 2,
              "softmax_stats must have at least 2 dimensions");
  TORCH_CHECK(partial_o.dim() == softmax_stats.dim(),
              "partial_o and softmax_stats must have same number of "
              "dimensions");

  int kv_lora_rank = partial_o.size(-1);
  TORCH_CHECK(partial_o.size(-2) == cp_size &&
                  softmax_stats.size(-2) == cp_size,
              "second-to-last dimension must equal cp_size");
  TORCH_CHECK(softmax_stats.size(-1) % 2 == 0 &&
                  softmax_stats.size(-1) >= 2,
              "softmax_stats last dimension must be divisible by 2 (float2)");
  bool allowVariableField1 = softmax_stats.size(-1) > 2;

  for (int i = 0; i < partial_o.dim() - 2; i++) {
    TORCH_CHECK(partial_o.size(i) == softmax_stats.size(i),
                "partial_o and softmax_stats must have matching dimensions "
                "except last two");
  }
  TORCH_CHECK(partial_o.size(-1) * partial_o.element_size() % 16 == 0,
              "partial_o must be aligned to 16 bytes");

  TORCH_CHECK(workspace.dim() == 2,
              "workspace must be 2D (strided across ranks)");
  TORCH_CHECK(workspace.size(0) == cp_size,
              "workspace must have cp_size rows");

  int entry_count = 1;
  for (int i = 0; i < partial_o.dim() - 2; i++) {
    entry_count *= partial_o.size(i);
  }

  torch::Tensor partial_o_3d =
      partial_o.reshape({entry_count, cp_size, kv_lora_rank});
  torch::Tensor softmax_stats_3d = softmax_stats.reshape(
      {entry_count, cp_size, softmax_stats.size(-1)});

  torch::Tensor partial_o_out = torch::empty_like(partial_o);
  torch::Tensor softmax_stats_out = torch::empty_like(softmax_stats);

  torch::Tensor partial_o_out_3d =
      partial_o_out.reshape({entry_count, cp_size, kv_lora_rank});
  torch::Tensor softmax_stats_out_3d = softmax_stats_out.reshape(
      {entry_count, cp_size, softmax_stats.size(-1)});

  vllm::kernels::HelixAllToAllParams params;

  params.sendFields[0].dataPtr =
      reinterpret_cast<uint8_t*>(partial_o_3d.data_ptr());
  params.sendFields[0].elementCount = kv_lora_rank;
  params.sendFields[0].elementSize = partial_o.element_size();
  params.sendFields[0].stride =
      partial_o_3d.stride(1) * partial_o.element_size();

  params.recvFields[0].dataPtr =
      reinterpret_cast<uint8_t*>(partial_o_out_3d.data_ptr());
  params.recvFields[0].elementCount = kv_lora_rank;
  params.recvFields[0].elementSize = partial_o.element_size();
  params.recvFields[0].stride =
      partial_o_out_3d.stride(1) * partial_o.element_size();

  params.sendFields[1].dataPtr =
      reinterpret_cast<uint8_t*>(softmax_stats_3d.data_ptr<float>());
  params.sendFields[1].elementCount = softmax_stats.size(-1);
  params.sendFields[1].elementSize = softmax_stats.element_size();
  params.sendFields[1].stride =
      softmax_stats_3d.stride(1) * softmax_stats.element_size();

  params.recvFields[1].dataPtr =
      reinterpret_cast<uint8_t*>(softmax_stats_out_3d.data_ptr<float>());
  params.recvFields[1].elementCount = softmax_stats.size(-1);
  params.recvFields[1].elementSize = softmax_stats.element_size();
  params.recvFields[1].stride =
      softmax_stats_out_3d.stride(1) * softmax_stats.element_size();

  params.entryCount = entry_count;
  params.workspace = reinterpret_cast<uint64_t*>(workspace.data_ptr());
  params.workspaceStrideInU64 = workspace.stride(0);

  params.cpRank = cp_rank;
  params.cpSize = cp_size;
  params.channelCount = 0;
  params.maxChannelCount =
      vllm::kernels::computeHelixMaxChannelCount(cp_size);

  auto stream = at::cuda::getCurrentCUDAStream();
  vllm::kernels::launchHelixAllToAll(params, allowVariableField1, stream);

  return std::make_tuple(partial_o_out, softmax_stats_out);
}

void initialize_helix_workspace(torch::Tensor workspace, int64_t cp_rank,
                                int64_t cp_size) {
  TORCH_CHECK(workspace.is_cuda(), "workspace must be on CUDA");
  TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Long,
              "workspace must be int64 (used as uint64)");
  TORCH_CHECK(workspace.dim() == 2, "workspace must be 2D");
  TORCH_CHECK(workspace.size(0) == cp_size,
              "workspace must have cp_size rows");
  TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size,
              "cp_rank must be in [0, cp_size)");

  auto stream = at::cuda::getCurrentCUDAStream();
  uint64_t* global_workspace_ptr =
      reinterpret_cast<uint64_t*>(workspace.data_ptr());
  uint64_t* local_workspace_ptr =
      reinterpret_cast<uint64_t*>(workspace[cp_rank].data_ptr());
  TORCH_CHECK(
      local_workspace_ptr ==
          global_workspace_ptr + cp_rank * workspace.stride(0),
      "local_workspace_ptr must be at the correct offset");
  vllm::kernels::initializeHelixWorkspace(local_workspace_ptr, cp_size,
                                          stream);
}

int64_t get_helix_workspace_size_per_rank(int64_t cp_size) {
  return static_cast<int64_t>(
      vllm::kernels::computeHelixWorkspaceSizePerRank(
          static_cast<int>(cp_size)));
}
