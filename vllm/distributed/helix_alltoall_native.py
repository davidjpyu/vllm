"""
Python wrapper for the native Helix all-to-all CUDA kernel.

Mirrors TRT-LLM's HelixAllToAllNative pattern:
  - Queries workspace size from C++
  - Allocates and caches workspace per (cp_rank, cp_size) group
  - Initializes workspace once
  - Provides run() to execute the native A2A and return output tensors

Phase 2: single-node only (plain CUDA device tensor).
Phase 3 will add MNNVL workspace for multi-node.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

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

        mgr = HelixAllToAllNative.get(cp_rank=0, cp_size=4)
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
    ) -> None:
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.workspace = workspace

    # ------------------------------------------------------------------
    # Factory / cache
    # ------------------------------------------------------------------

    @staticmethod
    def get(
        cp_rank: int,
        cp_size: int,
        device: Optional[torch.device] = None,
    ) -> "HelixAllToAllNative":
        """Get or create a manager for the given ``(cp_rank, cp_size)``."""
        key = (cp_rank, cp_size)
        if key not in HelixAllToAllNative._cache:
            if device is None:
                device = torch.device("cuda")

            ws_bytes = get_helix_workspace_size_per_rank(cp_size)
            ws_elems_per_rank = (ws_bytes + 7) // 8  # int64 elements

            logger.info(
                "Rank %d: allocating helix workspace — cp_size=%d, "
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

            ops = _get_ops()
            ops.initialize_helix_workspace(workspace, cp_rank, cp_size)
            torch.cuda.synchronize()

            HelixAllToAllNative._cache[key] = HelixAllToAllNative(
                cp_rank, cp_size, workspace
            )
            logger.info(
                "Rank %d: helix workspace initialized (cp_size=%d)",
                cp_rank,
                cp_size,
            )

        return HelixAllToAllNative._cache[key]

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
            f"workspace={self.workspace.shape})"
        )
