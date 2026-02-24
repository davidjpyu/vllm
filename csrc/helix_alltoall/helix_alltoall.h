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
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vllm {
namespace kernels {

struct HelixFieldInfo {
  uint8_t* dataPtr;
  int elementCount;
  int elementSize;
  int stride;
};

struct HelixAllToAllParams {
  HelixFieldInfo sendFields[2];
  HelixFieldInfo recvFields[2];
  int entryCount;
  uint64_t* workspace;
  size_t workspaceStrideInU64;
  int cpRank;
  int cpSize;
  int channelCount;
  int maxChannelCount;
};

int computeHelixMaxChannelCount(int cpSize, int smCount = 0);

size_t computeHelixWorkspaceSizePerRank(int cpSize);

void initializeHelixWorkspace(uint64_t* workspace, int cpSize,
                              cudaStream_t stream);

void launchHelixAllToAll(HelixAllToAllParams const& params,
                         bool allowVariableField1, cudaStream_t stream);

}  // namespace kernels
}  // namespace vllm
