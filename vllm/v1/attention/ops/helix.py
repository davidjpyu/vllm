# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helix parallelism operations for attention.

Helix uses All-to-All communication instead of AllGather+ReduceScatter
for context parallel attention, which can reduce communication overhead
for long-context scenarios.

Reference: https://arxiv.org/abs/2507.07120
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# ============================================================================
# Packed A2A Buffer Cache
# ============================================================================
# Helix uses a single packed A2A call per layer, fusing output and LSE into
# one tensor. Send/recv buffers are cached by shape to avoid per-call
# allocation. In vLLM continuous batching, B (decode token count) varies
# per step, but the scheduler stabilizes B at a few values during steady
# state. The cache typically holds 2-3 entries. Capped at _A2A_CACHE_LIMIT
# entries to bound GPU memory; oldest entries evicted on overflow.
_A2A_CACHE_LIMIT = 8
_a2a_buffers: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()


def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation for LSE-weighted combination.

    This is a pure PyTorch implementation for testing purposes.
    For GPU execution, use helix_lse_combine_triton instead.

    Args:
        outputs: Partial attention outputs [N, B, H, D]
                 N = number of KV shards (ranks)
                 B = batch size
                 H = number of heads
                 D = head dimension
        lses: Log-sum-exp values [N, B, H]
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H, D], and optionally global LSE [B, H]
    """
    N, B, H, D = outputs.shape

    # Handle NaN and inf in LSEs
    lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        torch.tensor(float("-inf"), device=lses.device, dtype=lses.dtype),
        lses,
    )

    # Compute max LSE for numerical stability
    lse_max, _ = lses.max(dim=0)  # [B, H]
    lse_max = torch.where(
        lse_max == float("-inf"),
        torch.zeros_like(lse_max),
        lse_max,
    )

    # Compute weights: softmax over the N dimension
    if is_lse_base_on_e:
        weights = torch.exp(lses - lse_max.unsqueeze(0))  # [N, B, H]
    else:
        weights = torch.pow(2.0, lses - lse_max.unsqueeze(0))  # [N, B, H]

    # Handle NaN weights
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

    # Normalize weights
    weight_sum = weights.sum(dim=0, keepdim=True)  # [1, B, H]
    weights = weights / weight_sum.clamp(min=1e-10)  # [N, B, H]

    # Weighted combination: sum over N dimension
    # outputs: [N, B, H, D], weights: [N, B, H] -> need to expand weights
    result = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, H, D]

    if return_lse:
        # Compute global LSE: logsumexp over N dimension
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


@triton.jit
def _helix_lse_combine_kernel(
    # Input pointers
    recv_output_ptr,
    recv_lse_ptr,
    # Output pointers
    out_ptr,
    out_lse_ptr,
    # Strides for recv_output [N, B, H_local, D]
    ro_stride_N,
    ro_stride_B,
    ro_stride_H,
    ro_stride_D,
    # Strides for recv_lse [N, B, H_local]
    rl_stride_N,
    rl_stride_B,
    rl_stride_H,
    # Strides for output [B, H_local, D]
    o_stride_B,
    o_stride_H,
    o_stride_D,
    # Constants
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """
    Triton kernel for Helix LSE-weighted combination.

    After All-to-All, each rank has:
    - recv_output [N, B, H_local, D]: partial outputs from all KV shards
    - recv_lse [N, B, H_local]: partial LSEs from all KV shards

    This kernel computes the weighted combination locally (no communication).

    Grid: (B, H_local)
    Each program handles one (batch, head) and processes all D elements.
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)

    # Base offset for this (batch, head)
    base_lse_offset = batch_idx * rl_stride_B + head_idx * rl_stride_H
    base_out_offset = batch_idx * ro_stride_B + head_idx * ro_stride_H

    # Step 1: Load all LSEs and compute weights
    # We need to load LSEs one by one and compute global LSE
    # First pass: find max LSE
    lse_max = -float("inf")
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        # Handle NaN and inf
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Second pass: compute sum of exp(lse - max)
    lse_sum = 0.0
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    # Compute global LSE (Triton kernel - keep if/else for clarity)
    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    # Step 2: Weighted combination across D dimension
    d_offsets = tl.arange(0, HEAD_DIM)

    # Initialize accumulator
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Third pass: weighted sum
    for n in tl.static_range(N):
        # Compute weight for this shard
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)

        # Load output for this shard and accumulate
        out_offsets = n * ro_stride_N + base_out_offset + d_offsets * ro_stride_D
        out_vals = tl.load(recv_output_ptr + out_offsets)
        acc += out_vals.to(tl.float32) * weight

    # Store result
    final_offsets = (
        batch_idx * o_stride_B + head_idx * o_stride_H + d_offsets * o_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    # Optional: store global LSE
    if RETURN_LSE:
        tl.store(out_lse_ptr + base_lse_offset, global_lse)


def helix_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated LSE-weighted combination for Helix.

    Args:
        recv_output: [N, B, H_local, D] - partial outputs from all KV shards
        recv_lse: [N, B, H_local] - partial LSEs from all KV shards
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H_local, D]
        If return_lse=True, also returns global_lse [B, H_local]
    """
    N, B, H_local, D = recv_output.shape

    # Allocate output tensors
    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )

    if return_lse:
        out_lse = torch.empty(
            (B, H_local), device=recv_lse.device, dtype=recv_lse.dtype
        )
    else:
        # Dummy tensor (not used, but kernel expects it)
        out_lse = torch.empty(1, device=recv_lse.device, dtype=recv_lse.dtype)

    # Get strides
    ro_stride_N, ro_stride_B, ro_stride_H, ro_stride_D = recv_output.stride()
    rl_stride_N, rl_stride_B, rl_stride_H = recv_lse.stride()
    o_stride_B, o_stride_H, o_stride_D = out.stride()

    # Launch kernel (grid must be 3-tuple)
    grid = (B, H_local, 1)

    _helix_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse,
        ro_stride_N,
        ro_stride_B,
        ro_stride_H,
        ro_stride_D,
        rl_stride_N,
        rl_stride_B,
        rl_stride_H,
        o_stride_B,
        o_stride_H,
        o_stride_D,
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )

    if return_lse:
        return out, out_lse
    return out


def _get_helix_a2a_backend() -> str:
    """Return the configured helix_a2a_backend, or "nccl" if unavailable."""
    try:
        from vllm.config import get_current_vllm_config
        return get_current_vllm_config().parallel_config.helix_a2a_backend
    except Exception:
        return "nccl"


def _helix_alltoall_nccl(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    kvp_group: GroupCoordinator,
    world_size: int,
    B: int,
    H_per_rank: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NCCL path: pack output+LSE into one tensor, all_to_all_single, unpack."""
    out_elem_size = local_output.element_size()
    lse_extra_elems = 4 // out_elem_size
    packed_D = D + lse_extra_elems

    cache_key = (
        B,
        H_per_rank,
        packed_D,
        world_size,
        local_output.dtype,
        str(local_output.device),
    )
    if cache_key in _a2a_buffers:
        _a2a_buffers.move_to_end(cache_key)
        send_packed, recv_packed = _a2a_buffers[cache_key]
    else:
        packed_shape = (world_size, B, H_per_rank, packed_D)
        send_packed = torch.empty(
            packed_shape, dtype=local_output.dtype, device=local_output.device
        )
        recv_packed = torch.empty(
            packed_shape, dtype=local_output.dtype, device=local_output.device
        )
        _a2a_buffers[cache_key] = (send_packed, recv_packed)
        logger.debug(
            "Helix A2A: allocated buffers for B=%d, packed_D=%d (cache entries: %d)",
            B,
            packed_D,
            len(_a2a_buffers),
        )
        while len(_a2a_buffers) > _A2A_CACHE_LIMIT:
            evicted_key, _ = _a2a_buffers.popitem(last=False)
            logger.debug(
                "Helix A2A: evicted buffer cache entry B=%d", evicted_key[0]
            )

    send_packed[:, :, :, :D].copy_(
        local_output.view(B, world_size, H_per_rank, D).permute(1, 0, 2, 3)
    )

    lse_permuted = (
        local_lse.view(B, world_size, H_per_rank).permute(1, 0, 2).contiguous()
    )

    if out_elem_size == 4:
        send_packed[:, :, :, D].copy_(lse_permuted)
    else:
        lse_reinterp = lse_permuted.view(local_output.dtype)
        lse_reinterp = lse_reinterp.view(
            world_size, B, H_per_rank, lse_extra_elems
        )
        send_packed[:, :, :, D:packed_D].copy_(lse_reinterp)

    dist.all_to_all_single(
        recv_packed.view(-1),
        send_packed.view(-1),
        group=kvp_group.device_group,
    )

    recv_output = recv_packed[:, :, :, :D].contiguous()

    if out_elem_size == 4:
        recv_lse = recv_packed[:, :, :, D].contiguous()
    else:
        recv_lse_raw = recv_packed[:, :, :, D:packed_D].contiguous()
        recv_lse = recv_lse_raw.view(
            world_size, B, H_per_rank * lse_extra_elems
        ).view(torch.float32)

    return recv_output, recv_lse


def _helix_alltoall_native(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    kvp_group: GroupCoordinator,
    world_size: int,
    B: int,
    H_per_rank: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native TRT-LLM-ported path: two-tensor A2A via custom CUDA kernel.

    Tensor layout contract for the native kernel:
      partial_o:     [..., cp_size, kv_lora_rank]  (half/bf16, contiguous)
      softmax_stats: [..., cp_size, 2]             (float32, contiguous)

    We reshape vLLM's [B, H, D] / [B, H] into these shapes, call the
    kernel, then reshape back to [N, B, H/N, D] / [N, B, H/N] for the
    Triton combine step.
    """
    from vllm.distributed.helix_alltoall_native import HelixAllToAllNative

    N = world_size
    cp_rank = kvp_group.rank_in_group
    cp_size = world_size
    entry_count = B * H_per_rank

    # --- Build partial_o: [entry_count, cp_size, D] ---
    # [B, H, D] -> [B, N, H/N, D] -> [B, H/N, N, D] -> [B*H/N, N, D]
    partial_o = (
        local_output.view(B, N, H_per_rank, D)
        .permute(0, 2, 1, 3)
        .reshape(entry_count, cp_size, D)
        .contiguous()
    )

    # --- Build softmax_stats: [entry_count, cp_size, 2] ---
    # [B, H] -> [B, N, H/N] -> [B, H/N, N] -> [B*H/N, N]
    lse_permuted = (
        local_lse.view(B, N, H_per_rank)
        .permute(0, 2, 1)
        .reshape(entry_count, cp_size)
        .contiguous()
    )
    # Pad to 2 floats per position: [entry_count, cp_size, 2]
    # First element is the LSE value, second is zero padding.
    softmax_stats = torch.zeros(
        entry_count, cp_size, 2,
        dtype=torch.float32,
        device=local_lse.device,
    )
    softmax_stats[:, :, 0] = lse_permuted

    # --- Run native kernel ---
    mgr = HelixAllToAllNative.get(
        cp_rank=cp_rank,
        cp_size=cp_size,
        cp_cpu_group=kvp_group.cpu_group,
    )
    partial_o_out, ss_out = mgr.run(partial_o, softmax_stats)

    # --- Reshape outputs for helix_lse_combine_triton ---
    # partial_o_out: [entry_count, cp_size, D] -> [B, H/N, N, D]
    #   -> [N, B, H/N, D]
    recv_output = (
        partial_o_out.view(B, H_per_rank, N, D)
        .permute(2, 0, 1, 3)
        .contiguous()
    )

    # ss_out: [entry_count, cp_size, 2] -> take first float -> [B*H/N, N]
    #   -> [B, H/N, N] -> [N, B, H/N]
    recv_lse = (
        ss_out[:, :, 0]
        .view(B, H_per_rank, N)
        .permute(2, 0, 1)
        .contiguous()
    )

    return recv_output, recv_lse


def helix_alltoall_lse_reduce(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    kvp_group: GroupCoordinator,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Helix-style attention output combination using All-to-All.

    Supports two backends (configured via ``helix_a2a_backend``):

    **NCCL (default):** Uses a single packed A2A call per layer, fusing
    output and LSE into one tensor [N,B,H/N,D+K] to minimize NCCL call
    count.

    **trtllm_native:** Uses the TRT-LLM-ported custom CUDA kernel with
    two separate tensors (partial_o and softmax_stats). The kernel uses
    LL128 + FIFO pipelining for lower latency, especially multi-node.

    Both paths produce ``recv_output [N, B, H/N, D]`` and
    ``recv_lse [N, B, H/N]``, then call ``helix_lse_combine_triton``
    for the final LSE-weighted combination.

    Args:
        local_output: Local attention output [B, H, D] where:
                      B = num_tokens, H = gathered_heads, D = kv_lora_rank
                      Each rank has output for the SAME tokens but computed
                      with DIFFERENT KV cache shards.
        local_lse: Local log-sum-exp values [B, H]
        kvp_group: GroupCoordinator for KV parallel communication
        return_lse: If True, also return the local portion of global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined attention output [B, H/N, D] (scattered along head dimension)
        If return_lse=True, also returns local_lse [B, H/N]
    """
    world_size = kvp_group.world_size

    if world_size == 1:
        if return_lse:
            return local_output, local_lse
        return local_output

    local_output = local_output.contiguous()
    local_lse = local_lse.contiguous()

    B, H, D = local_output.shape
    H_per_rank = H // world_size

    backend = _get_helix_a2a_backend()

    if backend == "trtllm_native":
        recv_output, recv_lse = _helix_alltoall_native(
            local_output, local_lse, kvp_group,
            world_size, B, H_per_rank, D,
        )
    else:
        recv_output, recv_lse = _helix_alltoall_nccl(
            local_output, local_lse, kvp_group,
            world_size, B, H_per_rank, D,
        )

    return helix_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )
