# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Workspace manager for FlashInfer's DCP all-to-all kernel.

Manages the workspace lifecycle (allocate → init → barrier → reuse) for
FlashInfer's ``dcp_a2a_alltoall``.  Replaces the prior vLLM-native
workspace manager (``dcp_alltoall_native.py``) and the manual MNNVL
allocation (``helix_mnnvl_workspace.py``).

Key simplifications over the native path:
  - No C++ ops compiled into vLLM — FlashInfer JIT-compiles the kernel.
  - No manual CUDA driver API calls for MNNVL — FlashInfer's
    ``MnnvlMemory`` handles it internally when ``mapping=`` is provided.
  - Workspace is reusable across calls without re-init (FIFO protocol).

Instances are cached by ``(cp_rank, cp_size)`` so workspace is allocated
and initialized exactly once per group configuration.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _to_torch(t: Any) -> torch.Tensor:
    """Convert a tvm_ffi.core.Tensor (or DLPack object) to torch.Tensor."""
    if isinstance(t, torch.Tensor):
        return t
    return torch.from_dlpack(t)


class DCPAllToAllFlashInfer:
    """Manages FlashInfer DCP workspace and executes the all-to-all kernel.

    Usage::

        mgr = DCPAllToAllFlashInfer.get(
            cp_rank=0, cp_size=4,
            cp_cpu_group=my_cp_cpu_group,
        )
        partial_o_out, ss_out = mgr.run(partial_o, softmax_stats)
    """

    _cache: dict[tuple[int, int], "DCPAllToAllFlashInfer"] = {}

    def __init__(
        self,
        cp_rank: int,
        cp_size: int,
        workspace: torch.Tensor,
        *,
        use_mnnvl: bool = False,
    ) -> None:
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.workspace = workspace
        self._use_mnnvl = use_mnnvl

    @staticmethod
    def get(
        cp_rank: int,
        cp_size: int,
        cp_cpu_group: Optional[dist.ProcessGroup] = None,
    ) -> "DCPAllToAllFlashInfer":
        """Get or create a manager for the given ``(cp_rank, cp_size)``.

        Args:
            cp_rank: Rank within the context-parallel group.
            cp_size: Size of the context-parallel group.
            cp_cpu_group: Optional **CPU** ProcessGroup for the CP ranks.
                When provided and MNNVL is requested, the workspace is
                allocated via FlashInfer's MnnvlMemory so it is visible
                across nodes.
        """
        key = (cp_rank, cp_size)
        if key not in DCPAllToAllFlashInfer._cache:
            workspace, use_mnnvl = DCPAllToAllFlashInfer._allocate(
                cp_rank, cp_size, cp_cpu_group,
            )

            from flashinfer.comm import dcp_a2a_init_workspace
            dcp_a2a_init_workspace(workspace, cp_rank, cp_size)

            if cp_cpu_group is not None:
                dist.barrier(group=cp_cpu_group)
            else:
                torch.cuda.synchronize()

            DCPAllToAllFlashInfer._cache[key] = DCPAllToAllFlashInfer(
                cp_rank, cp_size, workspace, use_mnnvl=use_mnnvl,
            )
            logger.info(
                "Rank %d: FlashInfer DCP workspace initialized "
                "(cp_size=%d, mnnvl=%s, shape=%s)",
                cp_rank, cp_size, use_mnnvl, list(workspace.shape),
            )

        return DCPAllToAllFlashInfer._cache[key]

    @staticmethod
    def _allocate(
        cp_rank: int,
        cp_size: int,
        cp_cpu_group: Optional[dist.ProcessGroup],
    ) -> tuple[torch.Tensor, bool]:
        """Allocate workspace, returning ``(workspace, used_mnnvl)``."""
        from flashinfer.comm import dcp_a2a_allocate_workspace

        use_mnnvl = DCPAllToAllFlashInfer._should_use_mnnvl(cp_cpu_group)

        if use_mnnvl:
            mapping, mnnvl_config = (
                DCPAllToAllFlashInfer._build_mnnvl_params(
                    cp_rank, cp_size, cp_cpu_group,
                )
            )
            workspace = dcp_a2a_allocate_workspace(
                cp_size, cp_rank,
                mapping=mapping,
                mnnvl_config=mnnvl_config,
            )
            logger.info(
                "Rank %d: MNNVL workspace allocated via FlashInfer — "
                "cp_size=%d, shape=%s",
                cp_rank, cp_size, list(workspace.shape),
            )
            return workspace, True

        workspace = dcp_a2a_allocate_workspace(cp_size, cp_rank)
        logger.info(
            "Rank %d: device workspace allocated via FlashInfer — "
            "cp_size=%d, shape=%s",
            cp_rank, cp_size, list(workspace.shape),
        )
        return workspace, False

    @staticmethod
    def _should_use_mnnvl(
        cp_cpu_group: Optional[dist.ProcessGroup],
    ) -> bool:
        """Decide whether to use MNNVL workspace allocation.

        - ``VLLM_DCP_USE_MNNVL=0`` → force device memory.
        - ``VLLM_DCP_USE_MNNVL=1`` → force MNNVL.
        - ``auto`` (default) → MNNVL when multi-node is detected.
        """
        env = os.environ.get("VLLM_DCP_USE_MNNVL", "auto").strip().lower()
        if env in ("0", "false", "no", "off"):
            return False
        if env in ("1", "true", "yes", "on"):
            return True

        if cp_cpu_group is None:
            return False
        try:
            from vllm.distributed.parallel_state import in_the_same_node_as
            same_node = in_the_same_node_as(cp_cpu_group, source_rank=0)
            return not all(same_node)
        except Exception:
            return False

    @staticmethod
    def _build_mnnvl_params(
        cp_rank: int,
        cp_size: int,
        cp_cpu_group: Optional[dist.ProcessGroup],
    ) -> tuple[Any, Any]:
        """Construct FlashInfer Mapping + MnnvlConfig from vLLM state."""
        from flashinfer.comm import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig, TorchDistBackend

        # Use tp_size=cp_size so all CP ranks share one MNNVL communicator.
        # MnnvlMemory.set_comm_from_config splits by
        #   color = pp_rank * cp_size + cp_rank, key = tp_rank
        # With cp_size=1, all ranks get color=0 (same group).
        # With tp_size=cp_size, each rank gets a unique key (0..N-1).
        mapping = Mapping(
            world_size=cp_size,
            rank=cp_rank,
            cp_size=1,
            tp_size=cp_size,
            pp_size=1,
        )

        mnnvl_config = None
        if cp_cpu_group is not None:
            mnnvl_config = MnnvlConfig(
                comm_backend=TorchDistBackend(group=cp_cpu_group),
            )

        return mapping, mnnvl_config

    def run(
        self,
        partial_o: torch.Tensor,
        softmax_stats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the FlashInfer all-to-all and return ``(recv_o, recv_stats)``.

        Args:
            partial_o: ``[..., cp_size, D]``, half or bf16.
            softmax_stats: ``[..., cp_size, S]``, float32, S >= 2 and even.

        Returns:
            Tuple of tensors with the same shapes/dtypes as inputs,
            containing the all-to-all exchanged data.
        """
        from flashinfer.comm import dcp_a2a_alltoall

        recv_o, recv_stats = dcp_a2a_alltoall(
            partial_o, softmax_stats,
            self.workspace, self.cp_rank, self.cp_size,
        )
        return _to_torch(recv_o), _to_torch(recv_stats)

    @staticmethod
    def clear_cache() -> None:
        """Drop all cached workspaces (useful for tests / shutdown)."""
        DCPAllToAllFlashInfer._cache.clear()

    @property
    def workspace_bytes_per_rank(self) -> int:
        from flashinfer.comm import dcp_a2a_workspace_size
        return dcp_a2a_workspace_size(self.cp_size)

    def __repr__(self) -> str:
        return (
            f"DCPAllToAllFlashInfer(cp_rank={self.cp_rank}, "
            f"cp_size={self.cp_size}, "
            f"mnnvl={self._use_mnnvl}, "
            f"workspace={self.workspace.shape})"
        )
