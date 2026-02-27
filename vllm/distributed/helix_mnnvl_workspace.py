"""
MNNVL workspace allocation for the Helix all-to-all kernel.

Multi-node NVLink (MNNVL) provides fabric-backed GPU memory where each
rank's workspace segment is mapped into a shared virtual address space,
enabling direct GPU-to-GPU access without NCCL in the hot path.

This module provides two strategies (tried in order):

1. **FlashInfer MnnvlMemory** — uses the same MNNVL allocator that vLLM
   already employs for MoE all-to-all.  Requires ``flashinfer`` with MNNVL
   support (``flashinfer.comm.mnnvl``).

2. **Device-memory fallback** — plain ``torch.zeros`` on CUDA.  Used when
   FlashInfer is not available or when running single-node.

The public entry point is :func:`allocate_helix_mnnvl_workspace`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_flashinfer_mnnvl_available: Optional[bool] = None


def is_flashinfer_mnnvl_available() -> bool:
    """Return True if FlashInfer's MNNVL allocator is importable."""
    global _flashinfer_mnnvl_available
    if _flashinfer_mnnvl_available is None:
        try:
            from flashinfer.comm import Mapping  # noqa: F401
            from flashinfer.comm.mnnvl import (  # noqa: F401
                MnnvlConfig,
                MnnvlMemory,
            )

            _flashinfer_mnnvl_available = True
        except (ImportError, ModuleNotFoundError):
            _flashinfer_mnnvl_available = False
    return _flashinfer_mnnvl_available


# ---------------------------------------------------------------------------
# Communicator that wraps a PyTorch ProcessGroup
# ---------------------------------------------------------------------------


def _make_helix_cp_communicator(
    cp_cpu_group: dist.ProcessGroup,
) -> Any:
    """Build a CommBackend wrapping *cp_cpu_group* with barrier support.

    If FlashInfer is available we subclass its ``CommBackend`` so that
    ``MnnvlMemory`` can use it directly.  Otherwise this is a no-op (the
    return value will not be used).
    """
    if not is_flashinfer_mnnvl_available():
        return None

    from flashinfer.comm.mnnvl import CommBackend

    class _HelixCpComm(CommBackend):
        """CommBackend backed by a PyTorch distributed CPU group."""

        def __init__(self, group: dist.ProcessGroup):
            self._group = group

        def Get_rank(self) -> int:
            return dist.get_rank(group=self._group)

        def Get_size(self) -> int:
            return dist.get_world_size(group=self._group)

        def allgather(self, data: Any):
            gathered = [None] * self.Get_size()
            dist.all_gather_object(gathered, data, group=self._group)
            return gathered

        def bcast(self, data: Any, root: int) -> Any:
            obj_list = [data]
            dist.broadcast_object_list(obj_list, src=root, group=self._group)
            return obj_list[0]

        def barrier(self) -> None:
            dist.barrier(group=self._group)

        def Split(self, color: int, key: int) -> "_HelixCpComm":
            return self

    return _HelixCpComm(cp_cpu_group)


# ---------------------------------------------------------------------------
# FlashInfer MNNVL allocation (Strategy A)
# ---------------------------------------------------------------------------


def _allocate_via_flashinfer(
    cp_rank: int,
    cp_size: int,
    ws_bytes_per_rank: int,
    cp_cpu_group: dist.ProcessGroup,
    gpus_per_node: int,
) -> Tuple[torch.Tensor, Any]:
    """Allocate MNNVL workspace using FlashInfer's ``MnnvlMemory``.

    Returns ``(workspace_tensor, mnnvl_handle)`` where *mnnvl_handle* is
    the ``MnnvlMemory`` instance that must be kept alive for the lifetime
    of the tensor.
    """
    from flashinfer.comm import Mapping
    from flashinfer.comm.mnnvl import MnnvlConfig, MnnvlMemory

    # Create a subclass so that helix CP state (comm, allocated_map, etc.)
    # is isolated from any other MnnvlMemory usage (e.g. MoE A2A).
    class _HelixCpMnnvlMemory(MnnvlMemory):
        pass

    comm_backend = _make_helix_cp_communicator(cp_cpu_group)

    config = MnnvlConfig(
        comm_backend=comm_backend,
        fabric_page_size=1 << 29,  # 512 MB
        allocation_granularity=0,  # auto-detect
    )

    mapping = Mapping(
        world_size=cp_size,
        rank=cp_rank,
        gpus_per_node=gpus_per_node,
        tp_size=cp_size,
    )

    _HelixCpMnnvlMemory.set_comm_from_config(mapping, config)

    memory = _HelixCpMnnvlMemory(mapping, ws_bytes_per_rank)
    workspace = memory.as_torch_strided_tensor(torch.int64)

    logger.info(
        "Rank %d: allocated MNNVL helix workspace via FlashInfer — "
        "shape=%s, stride=%d bytes/rank",
        cp_rank,
        list(workspace.shape),
        memory.rank_stride,
    )
    return workspace, memory


# ---------------------------------------------------------------------------
# Multi-node detection
# ---------------------------------------------------------------------------


def is_multi_node(cp_cpu_group: dist.ProcessGroup) -> bool:
    """Return True if *cp_cpu_group* spans more than one physical node."""
    from vllm.distributed.parallel_state import in_the_same_node_as

    same_node = in_the_same_node_as(cp_cpu_group, source_rank=0)
    return not all(same_node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def should_use_mnnvl(cp_cpu_group: Optional[dist.ProcessGroup]) -> bool:
    """Decide whether to use MNNVL workspace allocation.

    Decision logic (in order):

    1. ``VLLM_HELIX_USE_MNNVL=0``  → force device memory (always False).
    2. ``VLLM_HELIX_USE_MNNVL=1``  → force MNNVL (True if available).
    3. No env var (or ``auto``)     → True when *cp_cpu_group* spans
       multiple nodes **and** FlashInfer MNNVL is importable.
    """
    env = os.environ.get("VLLM_HELIX_USE_MNNVL", "auto").strip().lower()

    if env in ("0", "false", "no", "off"):
        return False

    if env in ("1", "true", "yes", "on"):
        if not is_flashinfer_mnnvl_available():
            logger.warning(
                "VLLM_HELIX_USE_MNNVL=1 but FlashInfer MNNVL is not available; "
                "falling back to device memory"
            )
            return False
        return True

    # auto: use MNNVL when multi-node and FlashInfer is available
    if cp_cpu_group is None:
        return False
    if not is_flashinfer_mnnvl_available():
        return False
    return is_multi_node(cp_cpu_group)


def allocate_helix_mnnvl_workspace(
    cp_rank: int,
    cp_size: int,
    ws_bytes_per_rank: int,
    cp_cpu_group: dist.ProcessGroup,
    gpus_per_node: Optional[int] = None,
) -> Tuple[torch.Tensor, Any]:
    """Allocate an MNNVL-backed helix workspace visible across nodes.

    Args:
        cp_rank: Rank within the context-parallel group.
        cp_size: Size of the context-parallel group.
        ws_bytes_per_rank: Workspace size in bytes per rank (from C++).
        cp_cpu_group: **CPU** ProcessGroup for the CP ranks.  Must be a
            non-NCCL group so that ``allgather`` of fabric handles works.
        gpus_per_node: GPUs per physical node (default: ``torch.cuda.device_count()``).

    Returns:
        ``(workspace_tensor, mnnvl_handle)`` — the tensor has shape
        ``[cp_size, ws_stride_in_i64]`` and dtype ``torch.int64``.
        *mnnvl_handle* must be kept alive (prevents deallocation).

    Raises:
        RuntimeError: If FlashInfer MNNVL is not available.
    """
    if not is_flashinfer_mnnvl_available():
        raise RuntimeError(
            "MNNVL workspace requested but FlashInfer MNNVL is not installed. "
            "Install flashinfer with MNNVL support or set VLLM_HELIX_USE_MNNVL=0."
        )

    if gpus_per_node is None:
        gpus_per_node = torch.cuda.device_count()

    return _allocate_via_flashinfer(
        cp_rank, cp_size, ws_bytes_per_rank, cp_cpu_group, gpus_per_node
    )
