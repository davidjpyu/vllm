"""
Python wrapper for the native Helix all-to-all CUDA kernel.

Mirrors TRT-LLM's HelixAllToAllNative pattern:
  - Queries workspace size from C++
  - Allocates and caches workspace per (cp_rank, cp_size) group
  - Initializes workspace once
  - Provides run() to execute the native A2A and return output tensors

Phase 2: single-node — plain CUDA device tensor.
Phase 3: multi-node — MNNVL workspace via FlashInfer (when available).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get_ops():
    """Return the torch.ops namespace that contains the helix ops.

    When built as part of vLLM (CMake with ENABLE_HELIX_ALLTOALL), the ops
    live under ``torch.ops._C``.  When JIT-compiled for standalone testing,
    they live on the module object returned by ``torch.utils.cpp_extension.load``.
    This helper always returns the vLLM built-in path; standalone tests bypass
    this module and call the JIT extension directly.
    """
    return torch.ops._C


def get_helix_workspace_size_per_rank(cp_size: int) -> int:
    """Return the workspace size **in bytes** for one rank."""
    return _get_ops().get_helix_workspace_size_per_rank(cp_size)


class HelixAllToAllNative:
    """Manages workspace allocation / init and executes the native A2A kernel.

    Usage::

        # Single-node (Phase 2 behaviour)
        mgr = HelixAllToAllNative.get(cp_rank=0, cp_size=4)

        # Multi-node — pass the CP CPU group for MNNVL allocation
        mgr = HelixAllToAllNative.get(cp_rank=0, cp_size=4,
                                       cp_cpu_group=my_cp_cpu_group)

        partial_o_out, ss_out = mgr.run(partial_o, softmax_stats)

    Instances are cached by ``(cp_rank, cp_size)`` so workspace is allocated
    and initialized exactly once per group configuration.
    """

    _cache: Dict[Tuple[int, int], "HelixAllToAllNative"] = {}

    def __init__(
        self,
        cp_rank: int,
        cp_size: int,
        workspace: torch.Tensor,
        *,
        mnnvl_handle: Any = None,
        use_mnnvl: bool = False,
    ) -> None:
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.workspace = workspace
        self._mnnvl_handle = mnnvl_handle  # prevent GC of MNNVL allocation
        self._use_mnnvl = use_mnnvl

    # ------------------------------------------------------------------
    # Factory / cache
    # ------------------------------------------------------------------

    @staticmethod
    def get(
        cp_rank: int,
        cp_size: int,
        device: Optional[torch.device] = None,
        cp_cpu_group: Optional[dist.ProcessGroup] = None,
    ) -> "HelixAllToAllNative":
        """Get or create a manager for the given ``(cp_rank, cp_size)``.

        Args:
            cp_rank: Rank within the context-parallel group.
            cp_size: Size of the context-parallel group.
            device: CUDA device for plain allocation (default: current).
            cp_cpu_group: Optional **CPU** ProcessGroup for the CP ranks.
                When provided and multi-node is detected (or
                ``VLLM_HELIX_USE_MNNVL=1``), the workspace is allocated
                via MNNVL so it is visible across nodes.  When ``None``
                the Phase 2 device-memory path is always used.
        """
        key = (cp_rank, cp_size)
        if key not in HelixAllToAllNative._cache:
            workspace, mnnvl_handle, use_mnnvl = (
                HelixAllToAllNative._allocate_workspace(
                    cp_rank, cp_size, device, cp_cpu_group
                )
            )

            ops = _get_ops()
            ops.initialize_helix_workspace(workspace, cp_rank, cp_size)
            torch.cuda.synchronize()

            if use_mnnvl and cp_cpu_group is not None:
                dist.barrier(group=cp_cpu_group)

            HelixAllToAllNative._cache[key] = HelixAllToAllNative(
                cp_rank,
                cp_size,
                workspace,
                mnnvl_handle=mnnvl_handle,
                use_mnnvl=use_mnnvl,
            )
            logger.info(
                "Rank %d: helix workspace initialized (cp_size=%d, mnnvl=%s)",
                cp_rank,
                cp_size,
                use_mnnvl,
            )

        return HelixAllToAllNative._cache[key]

    @staticmethod
    def _allocate_workspace(
        cp_rank: int,
        cp_size: int,
        device: Optional[torch.device],
        cp_cpu_group: Optional[dist.ProcessGroup],
    ) -> Tuple[torch.Tensor, Any, bool]:
        """Return ``(workspace, mnnvl_handle_or_None, used_mnnvl)``."""
        from vllm.distributed.helix_mnnvl_workspace import (
            allocate_helix_mnnvl_workspace,
            should_use_mnnvl,
        )

        ws_bytes = get_helix_workspace_size_per_rank(cp_size)

        if should_use_mnnvl(cp_cpu_group):
            assert cp_cpu_group is not None, (
                "MNNVL allocation requires a CP CPU process group"
            )
            logger.info(
                "Rank %d: allocating MNNVL helix workspace — cp_size=%d, "
                "%d bytes/rank (%.2f MB total)",
                cp_rank,
                cp_size,
                ws_bytes,
                ws_bytes * cp_size / (1024 * 1024),
            )
            workspace, handle = allocate_helix_mnnvl_workspace(
                cp_rank, cp_size, ws_bytes, cp_cpu_group
            )
            return workspace, handle, True

        # Phase 2 fallback: plain device memory
        if device is None:
            device = torch.device("cuda")

        ws_elems_per_rank = (ws_bytes + 7) // 8  # int64 elements
        logger.info(
            "Rank %d: allocating device helix workspace — cp_size=%d, "
            "%d bytes/rank (%d int64 elems/rank, %.2f MB total)",
            cp_rank,
            cp_size,
            ws_bytes,
            ws_elems_per_rank,
            ws_bytes * cp_size / (1024 * 1024),
        )
        workspace = torch.zeros(
            cp_size,
            ws_elems_per_rank,
            dtype=torch.long,
            device=device,
        )
        return workspace, None, False

    @staticmethod
    def clear_cache() -> None:
        """Drop all cached workspaces (useful for tests / shutdown)."""
        HelixAllToAllNative._cache.clear()

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def run(
        self,
        partial_o: torch.Tensor,
        softmax_stats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the native all-to-all and return ``(partial_o_out, ss_out)``.

        Args:
            partial_o: ``[..., cp_size, kv_lora_rank]``, half or bf16.
            softmax_stats: ``[..., cp_size, >=2]``, float32.

        Returns:
            Tuple of tensors with the same shapes / dtypes as the inputs,
            containing the all-to-all exchanged data.
        """
        ops = _get_ops()
        return ops.alltoall_helix_native(
            partial_o,
            softmax_stats,
            self.workspace,
            self.cp_rank,
            self.cp_size,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def workspace_bytes_per_rank(self) -> int:
        return get_helix_workspace_size_per_rank(self.cp_size)

    def __repr__(self) -> str:
        return (
            f"HelixAllToAllNative(cp_rank={self.cp_rank}, "
            f"cp_size={self.cp_size}, "
            f"mnnvl={self._use_mnnvl}, "
            f"workspace={self.workspace.shape})"
        )
