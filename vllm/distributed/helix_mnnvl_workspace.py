"""
MNNVL workspace allocation for the Helix all-to-all kernel (Strategy B).

Uses the CUDA driver API directly (via cuda-python bindings) to allocate
fabric-backed GPU memory visible across all ranks in the CP group.
This approach is ported from TRT-LLM's ``_mnnvl_utils.py``, with MPI
replaced by PyTorch distributed.

Architecture-dependent handle types (matching TRT-LLM):

- **aarch64 (GB200/NVL72):** ``CU_MEM_HANDLE_TYPE_FABRIC`` —
  serializable fabric handle blob, works **cross-node**.
- **x86_64 (H200 etc.):** ``CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`` —
  uses ``pidfd_open``/``pidfd_getfd`` syscalls, **intra-node only**.

**Why not FlashInfer MNNVL (Strategy A)?**
FlashInfer's ``MnnvlMemory`` internally uses ``pidfd`` for handle sharing
regardless of architecture.  This makes it fundamentally intra-node:
``pidfd_open(remote_pid)`` fails with ``ESRCH`` when processes run in
separate PID namespaces (e.g. Docker containers on different hosts).
Strategy B differentiates by architecture and uses FABRIC handles on
aarch64, enabling true multi-node MNNVL on GB200.

The public entry point is :func:`allocate_helix_mnnvl_workspace`.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_int,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint16,
    c_void_p,
    pointer,
)
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDA driver bindings (cuda-python)
# ---------------------------------------------------------------------------

_cuda_driver = None
_cuda_driver_checked = False


def _get_cuda_driver():
    """Lazily import the CUDA driver module from cuda-python."""
    global _cuda_driver, _cuda_driver_checked
    if _cuda_driver_checked:
        return _cuda_driver
    _cuda_driver_checked = True
    try:
        from cuda.bindings import driver as cuda_mod
        _cuda_driver = cuda_mod
    except ImportError:
        try:
            from cuda import cuda as cuda_mod
            _cuda_driver = cuda_mod
        except ImportError:
            _cuda_driver = None
    return _cuda_driver


def is_cuda_driver_available() -> bool:
    """Return True if cuda-python driver bindings are importable."""
    return _get_cuda_driver() is not None


def _check_cu_result(cu_func_ret):
    """Unwrap a CUDA driver API return tuple and raise on error."""
    cuda = _get_cuda_driver()
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver API error: {cu_result}")
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver API error: {cu_func_ret}")
        return None


# ---------------------------------------------------------------------------
# DLPack utilities (ported from TRT-LLM's _dlpack_utils.py)
#
# Wraps a raw GPU virtual-address pointer as a 2D strided PyTorch tensor
# so the rest of the code sees a normal ``torch.Tensor``.
# ---------------------------------------------------------------------------


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", c_uint8), ("bits", c_uint8), ("lanes", c_uint16)]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", c_int), ("device_id", c_int)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", c_void_p),
        ("device", _DLDevice),
        ("ndim", c_int),
        ("dtype", _DLDataType),
        ("shape", POINTER(c_int64)),
        ("strides", POINTER(c_int64)),
        ("byte_offset", c_size_t),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", c_void_p),
    ("deleter", CFUNCTYPE(None, POINTER(_DLManagedTensor))),
]


@CFUNCTYPE(None, POINTER(_DLManagedTensor))
def _no_op_deleter(dmt_ptr):
    pass


class _CapsuleKeepAlive:
    """Prevent GC of ctypes buffers referenced by the DLPack capsule."""

    def __init__(self, capsule, shape_array, stride_array, managed_tensor):
        self.capsule = capsule
        self._shape = shape_array
        self._stride = stride_array
        self._mt = managed_tensor


def _pack_strided_memory(
    ptr: int,
    segment_size: int,
    segment_stride: int,
    num_segments: int,
    dtype: torch.dtype,
    dev_id: int,
) -> torch.Tensor:
    """Wrap a raw GPU pointer as a 2-D strided PyTorch tensor via DLPack."""
    if dtype in (
        torch.float8_e5m2, torch.float8_e4m3fn, torch.bfloat16,
        torch.float16, torch.float32, torch.float64,
    ):
        bits = torch.finfo(dtype).bits
        type_code = 2
    elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        bits = torch.iinfo(dtype).bits
        type_code = 0
    elif dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        bits = torch.iinfo(dtype).bits
        type_code = 1
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype}")

    bpe = bits // 8
    ArrType = c_int64 * 2
    shape_arr = ArrType(num_segments, segment_size // bpe)
    stride_arr = ArrType(segment_stride // bpe, 1)

    device = _DLDevice(device_type=2, device_id=dev_id)
    dl_dtype = _DLDataType(code=type_code, bits=bits, lanes=1)

    dlt = _DLTensor()
    dlt.data = c_void_p(ptr)
    dlt.device = device
    dlt.ndim = 2
    dlt.dtype = dl_dtype
    dlt.shape = ctypes.cast(shape_arr, POINTER(c_int64))
    dlt.strides = ctypes.cast(stride_arr, POINTER(c_int64))
    dlt.byte_offset = 0

    mt = _DLManagedTensor()
    mt.dl_tensor = dlt
    mt.manager_ctx = None
    mt.deleter = _no_op_deleter

    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = c_void_p
    PyCapsule_New.argtypes = [c_void_p, ctypes.c_char_p, c_void_p]
    capsule_ptr = PyCapsule_New(pointer(mt), b"dltensor", None)
    capsule = ctypes.cast(capsule_ptr, ctypes.py_object).value

    keeper = _CapsuleKeepAlive(capsule, shape_arr, stride_arr, mt)
    tensor = torch.utils.dlpack.from_dlpack(keeper.capsule)
    tensor._dlpack_keeper = keeper
    return tensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FABRIC_PAGE_SIZE = 1 << 29  # 512 MB — matches TRT-LLM


# ---------------------------------------------------------------------------
# CUDA memory allocation helpers
# ---------------------------------------------------------------------------


def _get_allocation_prop(dev_id: int):
    """Build ``CUmemAllocationProp`` for the given device.

    On aarch64 (GB200): FABRIC handles (cross-node).
    On x86_64: POSIX_FILE_DESCRIPTOR handles (intra-node, uses pidfd).
    """
    cuda = _get_cuda_driver()

    location = cuda.CUmemLocation()
    location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = dev_id

    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED

    arch = platform.machine().lower()
    if "aarch64" in arch:
        prop.requestedHandleTypes = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        )
    else:
        prop.requestedHandleTypes = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )

    prop.location = location
    return prop


def _get_allocation_granularity(dev_id: int) -> int:
    cuda = _get_cuda_driver()
    prop = _get_allocation_prop(dev_id)
    option = cuda.CUmemAllocationGranularity_flags(
        cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    )
    return _check_cu_result(
        cuda.cuMemGetAllocationGranularity(prop=prop, option=option)
    )


# ---------------------------------------------------------------------------
# MNNVL memory handle
# ---------------------------------------------------------------------------


class HelixMnnvlHandle:
    """Holds MNNVL allocation state; prevent GC to keep memory mapped.

    Ported from TRT-LLM's ``MnnvlMemory``, with MPI replaced by
    PyTorch distributed.
    """

    def __init__(
        self,
        base_ptr: int,
        aligned_size: int,
        rank_stride: int,
        cp_size: int,
        dev_id: int,
        mem_handles: list,
    ):
        self.base_ptr = base_ptr
        self.aligned_size = aligned_size
        self.rank_stride = rank_stride
        self.cp_size = cp_size
        self.dev_id = dev_id
        self._mem_handles = mem_handles

    def as_torch_strided_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        """Return a 2-D ``[cp_size, elems_per_rank]`` strided tensor."""
        return _pack_strided_memory(
            self.base_ptr,
            self.aligned_size,
            self.rank_stride,
            self.cp_size,
            dtype,
            self.dev_id,
        )

    def __del__(self):
        if sys.is_finalizing():
            return
        cuda = _get_cuda_driver()
        if cuda is None:
            return
        for i in range(self.cp_size):
            rank_ptr = self.base_ptr + self.rank_stride * i
            try:
                _check_cu_result(cuda.cuMemUnmap(rank_ptr, self.aligned_size))
                _check_cu_result(cuda.cuMemRelease(self._mem_handles[i]))
            except Exception:
                pass
        try:
            device_ptr = cuda.CUdeviceptr(self.base_ptr)
            _check_cu_result(
                cuda.cuMemAddressFree(
                    device_ptr, self.cp_size * self.rank_stride
                )
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Core allocation logic
# ---------------------------------------------------------------------------


def _allgather_object(data: Any, group: dist.ProcessGroup) -> list:
    """Allgather a Python object across all ranks in the group."""
    world = dist.get_world_size(group=group)
    gathered: List[Any] = [None] * world
    dist.all_gather_object(gathered, data, group=group)
    return gathered


def _resolve_posix_fds(
    all_fds: List[int],
    all_pids: List[int],
    local_rank: int,
) -> List[int]:
    """Convert remote POSIX file descriptors to local FDs via pidfd syscalls.

    Only works intra-node (processes must share a PID namespace).
    Requires ``--cap-add=SYS_PTRACE`` in Docker containers.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    SYS_pidfd_open = 434
    SYS_pidfd_getfd = 438

    resolved: List[int] = []
    for i, (pid, fd) in enumerate(zip(all_pids, all_fds)):
        if i == local_rank:
            resolved.append(fd)
            continue

        pidfd = syscall(SYS_pidfd_open, pid, 0)
        if pidfd < 0:
            err = ctypes.get_errno()
            raise RuntimeError(
                f"pidfd_open({pid}) failed with errno {err}: "
                f"{os.strerror(err)}. "
                f"This syscall only works intra-node (same PID namespace). "
                f"For true cross-node MNNVL, aarch64/GB200 with FABRIC "
                f"handles is required."
            )

        remote_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0)
        if remote_fd < 0:
            err = ctypes.get_errno()
            msg = (
                f"pidfd_getfd(pidfd={pidfd}, fd={fd}) failed with "
                f"errno {err}: {os.strerror(err)}."
            )
            if err == 1:  # EPERM
                msg += (
                    " Permission denied. If running in a container, "
                    "try adding --cap-add=SYS_PTRACE to your docker run "
                    "command."
                )
            raise RuntimeError(msg)
        resolved.append(remote_fd)

    return resolved


def _allocate_mnnvl_memory(
    cp_rank: int,
    cp_size: int,
    ws_bytes_per_rank: int,
    cp_cpu_group: dist.ProcessGroup,
) -> HelixMnnvlHandle:
    """Allocate MNNVL memory using the CUDA driver API.

    Ported from TRT-LLM's ``MnnvlMemory.open_mnnvl_memory()``, with MPI
    replaced by PyTorch distributed (Gloo group).
    """
    cuda = _get_cuda_driver()

    # Ensure CUDA context is initialized
    _ = torch.empty(1, device="cuda")

    dev = _check_cu_result(cuda.cuCtxGetDevice())
    dev_id = int(dev)

    # Verify all ranks want the same size
    all_sizes = _allgather_object(ws_bytes_per_rank, cp_cpu_group)
    assert all(s == ws_bytes_per_rank for s in all_sizes), (
        f"Not all ranks requesting same workspace size: {all_sizes}"
    )

    # Align to granularity
    granularity = _get_allocation_granularity(dev_id)
    aligned_size = (
        (ws_bytes_per_rank + granularity - 1) // granularity * granularity
    )

    # Reserve virtual address space
    page_count = (aligned_size + FABRIC_PAGE_SIZE - 1) // FABRIC_PAGE_SIZE
    rank_stride = page_count * FABRIC_PAGE_SIZE
    address_size = rank_stride * cp_size

    base_ptr = _check_cu_result(
        cuda.cuMemAddressReserve(address_size, FABRIC_PAGE_SIZE, 0, 0)
    )
    base_ptr_int = int(base_ptr)

    logger.info(
        "Rank %d: MNNVL vaddr reserved — base=0x%x, stride=%d, "
        "total=%d bytes, arch=%s",
        cp_rank,
        base_ptr_int,
        rank_stride,
        address_size,
        platform.machine(),
    )

    # Allocate local physical memory
    prop = _get_allocation_prop(dev_id)
    local_handle = _check_cu_result(
        cuda.cuMemCreate(aligned_size, prop, flags=0)
    )

    # Export local handle
    exported = _check_cu_result(
        cuda.cuMemExportToShareableHandle(
            local_handle, prop.requestedHandleTypes, 0
        )
    )

    # Allgather handles across ranks
    is_fabric = (
        prop.requestedHandleTypes
        == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    )

    if is_fabric:
        all_handle_data = _allgather_object(exported.data, cp_cpu_group)
    else:
        all_fds = _allgather_object(exported, cp_cpu_group)
        all_pids = _allgather_object(os.getpid(), cp_cpu_group)
        all_handle_data = _resolve_posix_fds(all_fds, all_pids, cp_rank)

    # Map all ranks' memory into the shared virtual address space
    madesc = cuda.CUmemAccessDesc()
    madesc.location = prop.location
    madesc.flags = (
        cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    )

    mem_handles: List[Any] = []
    for i, remote_data in enumerate(all_handle_data):
        rank_ptr = base_ptr_int + rank_stride * i

        if i == cp_rank:
            _check_cu_result(
                cuda.cuMemMap(rank_ptr, aligned_size, 0, local_handle, 0)
            )
            mem_handles.append(local_handle)
        else:
            imported = _check_cu_result(
                cuda.cuMemImportFromShareableHandle(
                    remote_data, prop.requestedHandleTypes
                )
            )
            _check_cu_result(
                cuda.cuMemMap(rank_ptr, aligned_size, 0, imported, 0)
            )
            mem_handles.append(imported)

        _check_cu_result(
            cuda.cuMemSetAccess(rank_ptr, aligned_size, [madesc], 1)
        )

    return HelixMnnvlHandle(
        base_ptr=base_ptr_int,
        aligned_size=aligned_size,
        rank_stride=rank_stride,
        cp_size=cp_size,
        dev_id=dev_id,
        mem_handles=mem_handles,
    )


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


def _is_fabric_capable() -> bool:
    """Return True if the platform uses FABRIC handles (aarch64/GB200).

    On x86_64, MNNVL uses POSIX FD handles + pidfd syscalls which only
    work intra-node.  Cross-node MNNVL requires FABRIC handles, which
    are only available on aarch64 (GB200/NVL72) where NVLink physically
    spans nodes.
    """
    return "aarch64" in platform.machine().lower()


def should_use_mnnvl(cp_cpu_group: Optional[dist.ProcessGroup]) -> bool:
    """Decide whether to use MNNVL workspace allocation.

    Decision logic (in order):

    1. ``VLLM_HELIX_USE_MNNVL=0``  → force device memory (always False).
    2. ``VLLM_HELIX_USE_MNNVL=1``  → force MNNVL (True if cuda-python
       driver bindings are available).  Use this for intra-node MNNVL
       testing on x86_64 (H200).
    3. No env var (or ``auto``)     → True when *cp_cpu_group* spans
       multiple nodes **and** cuda-python is importable **and** the
       platform is aarch64 (GB200).  On x86_64, multi-node MNNVL is
       not possible because NVLink does not span nodes on H200.
    """
    env = os.environ.get("VLLM_HELIX_USE_MNNVL", "auto").strip().lower()

    if env in ("0", "false", "no", "off"):
        return False

    if env in ("1", "true", "yes", "on"):
        if not is_cuda_driver_available():
            logger.warning(
                "VLLM_HELIX_USE_MNNVL=1 but cuda-python driver bindings "
                "are not available; falling back to device memory"
            )
            return False
        return True

    # auto: use MNNVL when multi-node + cuda driver + aarch64 (GB200)
    if cp_cpu_group is None:
        return False
    if not is_cuda_driver_available():
        return False
    if not is_multi_node(cp_cpu_group):
        return False
    if not _is_fabric_capable():
        logger.info(
            "Multi-node detected on x86_64 — MNNVL not available "
            "(no cross-node NVLink on H200). Using device memory. "
            "Set VLLM_HELIX_USE_MNNVL=1 to force intra-node MNNVL."
        )
        return False
    return True


def allocate_helix_mnnvl_workspace(
    cp_rank: int,
    cp_size: int,
    ws_bytes_per_rank: int,
    cp_cpu_group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, HelixMnnvlHandle]:
    """Allocate an MNNVL-backed helix workspace visible across nodes.

    Uses the CUDA driver API directly (matching TRT-LLM's approach):

    - aarch64 (GB200): ``CU_MEM_HANDLE_TYPE_FABRIC`` → cross-node
    - x86_64: ``CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`` → intra-node

    Args:
        cp_rank: Rank within the context-parallel group.
        cp_size: Size of the context-parallel group.
        ws_bytes_per_rank: Workspace size in bytes per rank (from C++).
        cp_cpu_group: **CPU** ProcessGroup for the CP ranks.  Must be a
            non-NCCL group so that ``allgather`` of handle data works.

    Returns:
        ``(workspace_tensor, mnnvl_handle)`` — the tensor has shape
        ``[cp_size, aligned_size_in_i64]`` and dtype ``torch.int64``.
        *mnnvl_handle* must be kept alive (prevents deallocation).

    Raises:
        RuntimeError: If cuda-python driver bindings are not available.
    """
    if not is_cuda_driver_available():
        raise RuntimeError(
            "MNNVL workspace requested but cuda-python driver bindings "
            "are not installed. Install cuda-python "
            "(pip install cuda-python) or set VLLM_HELIX_USE_MNNVL=0."
        )

    handle = _allocate_mnnvl_memory(
        cp_rank, cp_size, ws_bytes_per_rank, cp_cpu_group
    )
    workspace = handle.as_torch_strided_tensor(torch.int64)

    logger.info(
        "Rank %d: MNNVL helix workspace allocated — shape=%s, "
        "stride=%d bytes/rank, arch=%s",
        cp_rank,
        list(workspace.shape),
        handle.rank_stride,
        platform.machine(),
    )

    return workspace, handle
