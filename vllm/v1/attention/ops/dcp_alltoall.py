# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DCP All-to-All communication backend for attention.

Provides All-to-All (A2A) communication as an alternative to
AllGather + ReduceScatter (AG+RS) for Decode Context Parallel (DCP).
Instead of gathering the full Q tensor and scattering partial outputs,
A2A exchanges partial attention outputs and their LSE values across
ranks, then combines them with exact LSE-weighted reduction.

This reduces the number of NCCL calls per attention layer from 3
(AG for Q, AG for K metadata, RS for output) to 2 (A2A for output,
A2A for LSE), lowering per-step communication overhead for long-context
decode where NCCL latency is a significant fraction of step time.

Usage:
    vllm serve model --tp 16 --dcp 16 --dcp-comm-backend a2a

Reference: https://arxiv.org/abs/2507.07120
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

# A/B-test toggle: if DCP_A2A_RESHAPE_OLD=1, use the original vllm
# [B*H_per_rank, N, D] reshape instead of the TRT-LLM-style packed
# [B, N, H_per_rank * D] layout. Read once at import.
_USE_OLD_RESHAPE = os.environ.get("DCP_A2A_RESHAPE_OLD", "0") == "1"

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops.common import CPTritonContext


def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation for LSE-weighted combination.

    This is a pure PyTorch implementation used for testing and validation.
    For GPU execution, use dcp_lse_combine_triton instead.

    Args:
        outputs: Partial attention outputs [N, B, H, D]
                 N = number of KV shards (ranks)
                 B = batch size (num_tokens)
                 H = number of heads per rank
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
    result = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, H, D]

    if return_lse:
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


@triton.jit
def _dcp_lse_combine_kernel(
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
    Triton kernel for LSE-weighted combination of partial attention outputs.

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

    # First pass: find max LSE for numerical stability
    lse_max = -float("inf")
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
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

    # Compute global LSE
    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    # Third pass: weighted combination across D dimension
    d_offsets = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for n in tl.static_range(N):
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

        out_offsets = n * ro_stride_N + base_out_offset + d_offsets * ro_stride_D
        out_vals = tl.load(recv_output_ptr + out_offsets)
        acc += out_vals.to(tl.float32) * weight

    # Store result
    final_offsets = (
        batch_idx * o_stride_B + head_idx * o_stride_H + d_offsets * o_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    if RETURN_LSE:
        tl.store(out_lse_ptr + base_lse_offset, global_lse)


def dcp_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated LSE-weighted combination for DCP A2A.

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

    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )

    if return_lse:
        out_lse = torch.empty(
            (B, H_local), device=recv_lse.device, dtype=recv_lse.dtype
        )
    else:
        out_lse = torch.empty(1, device=recv_lse.device, dtype=recv_lse.dtype)

    ro_stride_N, ro_stride_B, ro_stride_H, ro_stride_D = recv_output.stride()
    rl_stride_N, rl_stride_B, rl_stride_H = recv_lse.stride()
    o_stride_B, o_stride_H, o_stride_D = out.stride()

    grid = (B, H_local, 1)

    _dcp_lse_combine_kernel[grid](
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


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Combine partial attention outputs across DCP ranks using All-to-All.

    Each rank holds attention output for all heads but only a local shard
    of the KV cache. This function:
    1. Exchanges partial outputs across ranks via All-to-All
    2. Exchanges LSE values via All-to-All
    3. Combines them with exact LSE-weighted reduction (Triton kernel)

    Tensor flow:
        Input:  cp_attn_out [B, H, D] - all heads, local KV shard
        Reshape: [N, B, H/N, D] - split heads across ranks
        A2A:    Two all_to_all_single calls (output and LSE)
        Combine: recv [N, B, H/N, D] + lse [N, B, H/N] -> [B, H/N, D]

    Args:
        cp_attn_out: [B, H, D] where B=num_tokens, H=total_heads, D=head_dim
        cp_attn_lse: [B, H] log-sum-exp values (fp32)
        cp_group: GroupCoordinator for DCP communication
        ctx: CPTritonContext (unused, for signature compatibility)
        return_lse: If True, also return the combined global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H/N, D] (head-scattered)
        If return_lse=True, also returns global_lse [B, H/N]
    """
    world_size = cp_group.world_size

    if world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    local_output = cp_attn_out.contiguous()
    local_lse = cp_attn_lse.contiguous()

    B, H, D = local_output.shape
    H_per_rank = H // world_size

    _backend = _get_dcp_a2a_backend()
    if _backend == "flashinfer":
        recv_output, recv_lse = _alltoall_flashinfer(
            local_output, local_lse, cp_group, B, world_size, H_per_rank, D
        )
    else:
        # Reshape for All-to-All: [B, H, D] -> [N, B, H/N, D]
        # Split heads into N chunks, each destined for a different rank
        send_output = (
            local_output.view(B, world_size, H_per_rank, D)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        recv_output = torch.empty_like(send_output)

        # Same for LSE: [B, H] -> [N, B, H/N]
        send_lse = (
            local_lse.view(B, world_size, H_per_rank).permute(1, 0, 2).contiguous()
        )
        recv_lse = torch.empty_like(send_lse)

        # All-to-All for partial attention outputs and LSE values (async overlap)
        work_output = dist.all_to_all_single(
            recv_output.view(-1),
            send_output.view(-1),
            group=cp_group.device_group,
            async_op=True,
        )
        work_lse = dist.all_to_all_single(
            recv_lse.view(-1),
            send_lse.view(-1),
            group=cp_group.device_group,
            async_op=True,
        )
        work_output.wait()
        work_lse.wait()

    # LSE-weighted combination via Triton kernel (local, no communication)
    return dcp_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )


# Module-level cache of the DCP A2A backend choice. Set by
# ``gpu_worker.py`` at workspace pre-init (where vllm_config is reliably
# available), and read by the dispatcher during model forward — V1's
# async scheduling path doesn't enter ``set_current_vllm_config()`` for
# every forward, so ``get_current_vllm_config()`` raises and the original
# fallback silently routed every A2A call to NCCL even when the user
# asked for FlashInfer.
_DCP_A2A_BACKEND: str | None = None


def set_dcp_a2a_backend(backend: str) -> None:
    """Cache the DCP A2A backend on this worker process.

    Called once from ``gpu_worker.py`` after parsing the parallel
    config, so the dispatcher can route correctly without depending on
    ``get_current_vllm_config()`` during forward.
    """
    global _DCP_A2A_BACKEND
    _DCP_A2A_BACKEND = backend


def _get_dcp_a2a_backend() -> str:
    """Return the DCP A2A backend (``"nccl"`` or ``"flashinfer"``)."""
    if _DCP_A2A_BACKEND is not None:
        return _DCP_A2A_BACKEND
    try:
        from vllm.config import get_current_vllm_config
        return get_current_vllm_config().parallel_config.dcp_a2a_backend
    except Exception:
        return "nccl"


def _alltoall_flashinfer(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    cp_group: "GroupCoordinator",
    B: int,
    world_size: int,
    H_per_rank: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer DCP A2A path (single fused LL128 + MNNVL kernel).

    FlashInfer's ``decode_cp_a2a_alltoall`` expects::

        partial_o:     [..., cp_size, D]            half/bfloat16
        softmax_stats: [..., cp_size, S]            float32, S>=2 even

    We reshape vLLM's ``[B, H, D]`` / ``[B, H]`` into those shapes, run
    the kernel, then reshape back to ``[N, B, H/N, D]`` / ``[N, B, H/N]``
    so the Triton LSE-combine kernel below works unchanged.

    The kernel only shuffles bytes — it does not interpret the
    ``softmax_stats`` payload semantically, so packing the LSE into
    ``stats[..., 0]`` (with a trailing zero in slot 1) round-trips
    cleanly regardless of ``is_lse_base_on_e``.
    """
    from vllm.distributed.dcp_alltoall_flashinfer import DCPAllToAllFlashInfer

    N = world_size

    if _USE_OLD_RESHAPE:
        # Original vllm reshape: [B*H_per_rank, N, D]. Higher per-call
        # entry count (H_per_rank× more) and an extra .contiguous() copy
        # vs the TRT-LLM-style packed layout below. Kept behind an env
        # var for A/B comparison.
        entry_count = B * H_per_rank
        partial_o = (
            local_output.view(B, N, H_per_rank, D)
            .permute(0, 2, 1, 3)
            .reshape(entry_count, N, D)
            .contiguous()
        )
        lse_permuted = (
            local_lse.view(B, N, H_per_rank)
            .permute(0, 2, 1)
            .reshape(entry_count, N)
            .contiguous()
        )
        softmax_stats = torch.zeros(
            entry_count, N, 2, dtype=torch.float32, device=local_lse.device
        )
        softmax_stats[..., 0] = lse_permuted

        mgr = DCPAllToAllFlashInfer.get(
            cp_rank=cp_group.rank_in_group,
            cp_size=N,
            cp_cpu_group=cp_group.cpu_group,
        )
        partial_o_out, ss_out = mgr.run(partial_o, softmax_stats)

        recv_output = (
            partial_o_out.view(B, H_per_rank, N, D)
            .permute(2, 0, 1, 3)
            .contiguous()
        )
        recv_lse = (
            ss_out[..., 0]
            .view(B, H_per_rank, N)
            .permute(2, 0, 1)
            .contiguous()
        )
        return recv_output, recv_lse

    # Default: TRT-LLM-style packed layout. Pack heads-in-partition into
    # the last dim so the kernel's first dim is num_tokens (B). Matches
    # what TRT-LLM's ``_attn_forward_gen`` passes:
    #
    #   partial_o.view(num_tokens, cp_size, num_heads_tp_cp * value_dim)
    #
    # The head order is [CP0_heads | CP1_heads | ... | CPN_heads] along
    # dim 1 (preserved by the prior AllGather(dim=1)), so a plain view to
    # [B, N, H_per_rank * D] is correct without a permute.
    partial_o = local_output.view(B, N, H_per_rank * D)

    # softmax_stats: pack [lse, 0] for each (token, peer, h_per_rank) so the
    # last dim has stride 2 (S=2). lse layout follows partial_o: [B, N, H/N].
    lse_3d = local_lse.view(B, N, H_per_rank)
    zeros_3d = torch.zeros_like(lse_3d)
    softmax_stats = (
        torch.stack([lse_3d, zeros_3d], dim=-1)  # [B, N, H/N, 2]
        .view(B, N, H_per_rank * 2)
        .contiguous()
    )

    mgr = DCPAllToAllFlashInfer.get(
        cp_rank=cp_group.rank_in_group,
        cp_size=N,
        cp_cpu_group=cp_group.cpu_group,
    )
    partial_o_out, ss_out = mgr.run(partial_o, softmax_stats)

    # Output permute back to [N, B, H/N, D] for downstream triton.
    recv_output = (
        partial_o_out.view(B, N, H_per_rank, D)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    # ss_out: [B, N, H/N * 2] -> [B, N, H/N, 2] -> take lse slot 0 -> [N, B, H/N]
    recv_lse = (
        ss_out.view(B, N, H_per_rank, 2)[..., 0]
        .permute(1, 0, 2)
        .contiguous()
    )
    return recv_output, recv_lse
