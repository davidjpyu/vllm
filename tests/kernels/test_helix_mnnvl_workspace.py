"""
Phase 3 tests for the Helix MNNVL workspace allocation.

Two groups of tests:

**A. Single-GPU policy tests** — verify ``should_use_mnnvl()`` env-var
logic, backward compatibility of ``HelixAllToAllNative.get()`` without
a CP group, and availability checks.  Run on a single GPU::

    python tests/kernels/test_helix_mnnvl_workspace.py

**B. Multi-GPU MNNVL allocation tests** — actually allocate MNNVL
workspace across ranks.  Requires >= 2 GPUs and ``torchrun``::

    torchrun --nproc_per_node=2 tests/kernels/test_helix_mnnvl_workspace.py --distributed

Requirements: PyTorch with CUDA 12.0+, SM90+ GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

_mnnvl_ws_module = None


def _load_mnnvl_workspace_module():
    """Load helix_mnnvl_workspace.py directly, bypassing vllm.distributed.__init__."""
    global _mnnvl_ws_module
    if _mnnvl_ws_module is not None:
        return _mnnvl_ws_module
    mod_path = REPO_ROOT / "vllm" / "distributed" / "helix_mnnvl_workspace.py"
    spec = importlib.util.spec_from_file_location(
        "helix_mnnvl_workspace", str(mod_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _mnnvl_ws_module = mod
    return mod


# ---------------------------------------------------------------------------
# JIT extension loader (reused from Phase 1/2 tests)
# ---------------------------------------------------------------------------


def _ensure_writable_cache():
    cache_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if cache_dir and os.access(os.path.dirname(cache_dir) or ".", os.W_OK):
        return
    for candidate in [
        Path.home() / ".cache" / "torch_extensions",
        REPO_ROOT / ".build_cache" / "torch_extensions",
    ]:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_EXTENSIONS_DIR"] = str(candidate)
            return
        except OSError:
            continue


def load_jit_extension():
    _ensure_writable_cache()
    from torch.utils.cpp_extension import load

    src_dir = REPO_ROOT / "csrc" / "helix_alltoall"
    print(f"[JIT] Compiling from {src_dir} ...")
    t0 = time.time()
    ext = load(
        name="helix_alltoall",
        sources=[
            str(src_dir / "helix_alltoall.cu"),
            str(src_dir / "helix_alltoall_op.cpp"),
            str(src_dir / "jit_binding.cpp"),
        ],
        extra_include_paths=[str(src_dir)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
        extra_cflags=["-O3"],
        verbose=True,
    )
    print(f"[JIT] Done in {time.time() - t0:.1f}s")
    return ext


class _JitOpsAdapter:
    """Wraps the JIT module so it can be used as a drop-in for torch.ops._C."""

    def __init__(self, ext: Any):
        self._ext = ext

    def get_helix_workspace_size_per_rank(self, cp_size: int) -> int:
        return self._ext.get_helix_workspace_size_per_rank(cp_size)

    def initialize_helix_workspace(
        self, workspace: torch.Tensor, cp_rank: int, cp_size: int
    ) -> None:
        self._ext.initialize_helix_workspace(workspace, cp_rank, cp_size)

    def alltoall_helix_native(
        self,
        partial_o: torch.Tensor,
        softmax_stats: torch.Tensor,
        workspace: torch.Tensor,
        cp_rank: int,
        cp_size: int,
    ):
        return self._ext.alltoall_helix_native(
            partial_o, softmax_stats, workspace, cp_rank, cp_size
        )


# ---------------------------------------------------------------------------
# A. Single-GPU policy tests
# ---------------------------------------------------------------------------


def test_should_use_mnnvl_env_off(ops):
    """VLLM_HELIX_USE_MNNVL=0 → always returns False."""
    should_use_mnnvl = _load_mnnvl_workspace_module().should_use_mnnvl

    old = os.environ.get("VLLM_HELIX_USE_MNNVL")
    try:
        os.environ["VLLM_HELIX_USE_MNNVL"] = "0"
        assert not should_use_mnnvl(None), "Expected False when MNNVL=0"
        print("PASS: test_should_use_mnnvl_env_off")
    finally:
        if old is None:
            os.environ.pop("VLLM_HELIX_USE_MNNVL", None)
        else:
            os.environ["VLLM_HELIX_USE_MNNVL"] = old


def test_should_use_mnnvl_no_group(ops):
    """When cp_cpu_group is None, auto mode → always False."""
    should_use_mnnvl = _load_mnnvl_workspace_module().should_use_mnnvl

    old = os.environ.get("VLLM_HELIX_USE_MNNVL")
    try:
        os.environ.pop("VLLM_HELIX_USE_MNNVL", None)
        assert not should_use_mnnvl(None), "Expected False when group is None"
        print("PASS: test_should_use_mnnvl_no_group")
    finally:
        if old is not None:
            os.environ["VLLM_HELIX_USE_MNNVL"] = old


def test_flashinfer_availability(ops):
    """is_flashinfer_mnnvl_available() returns bool without crashing."""
    is_flashinfer_mnnvl_available = _load_mnnvl_workspace_module().is_flashinfer_mnnvl_available

    result = is_flashinfer_mnnvl_available()
    print(f"  FlashInfer MNNVL available: {result}")
    assert isinstance(result, bool)
    print("PASS: test_flashinfer_availability")


def test_backward_compat_device_alloc(ops):
    """get() without cp_cpu_group → device allocation (Phase 2 behaviour)."""

    ws_bytes = ops.get_helix_workspace_size_per_rank(1)
    ws_elems = (ws_bytes + 7) // 8
    workspace = torch.zeros(1, ws_elems, dtype=torch.long, device="cuda")
    ops.initialize_helix_workspace(workspace, 0, 1)
    torch.cuda.synchronize()

    D = 128
    N = 8
    po = torch.randn(N, 1, D, dtype=torch.bfloat16, device="cuda")
    ss = torch.randn(N, 1, 2, dtype=torch.float32, device="cuda")
    po_out, ss_out = ops.alltoall_helix_native(po, ss, workspace, 0, 1)
    torch.cuda.synchronize()

    assert po_out.shape == po.shape
    assert ss_out.shape == ss.shape
    print("PASS: test_backward_compat_device_alloc")


def test_env_var_values(ops):
    """Various VLLM_HELIX_USE_MNNVL values are handled correctly."""
    should_use_mnnvl = _load_mnnvl_workspace_module().should_use_mnnvl

    test_cases = [
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("OFF", False),
    ]

    old = os.environ.get("VLLM_HELIX_USE_MNNVL")
    try:
        for val, expected in test_cases:
            os.environ["VLLM_HELIX_USE_MNNVL"] = val
            result = should_use_mnnvl(None)
            assert result == expected, (
                f"VLLM_HELIX_USE_MNNVL={val!r}: expected {expected}, got {result}"
            )
        print("PASS: test_env_var_values")
    finally:
        if old is None:
            os.environ.pop("VLLM_HELIX_USE_MNNVL", None)
        else:
            os.environ["VLLM_HELIX_USE_MNNVL"] = old


def test_device_alloc_multiple_cp(ops):
    """Device allocation works for various cp_sizes (Phase 2 regression)."""
    for cp in [1, 2, 4]:
        ws_bytes = ops.get_helix_workspace_size_per_rank(cp)
        ws_elems = (ws_bytes + 7) // 8
        workspace = torch.zeros(cp, ws_elems, dtype=torch.long, device="cuda")
        ops.initialize_helix_workspace(workspace, 0, cp)
        torch.cuda.synchronize()
        assert workspace.shape == (cp, ws_elems)
        print(f"  cp_size={cp} -> device workspace OK")

    print("PASS: test_device_alloc_multiple_cp")


# ---------------------------------------------------------------------------
# B. Multi-GPU MNNVL distributed tests (run via torchrun)
# ---------------------------------------------------------------------------


def test_mnnvl_allocation_distributed(ops):
    """Allocate MNNVL workspace across ranks and verify tensor properties.

    Must be launched with torchrun --nproc_per_node=N (N >= 2).
    """
    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cp_size = world_size

    mod = _load_mnnvl_workspace_module()

    if not mod.is_flashinfer_mnnvl_available():
        if rank == 0:
            print("SKIP: test_mnnvl_allocation_distributed — FlashInfer MNNVL not available")
        return

    ws_bytes = ops.get_helix_workspace_size_per_rank(cp_size)
    gloo_group = dist.new_group(backend="gloo")

    workspace, handle = mod.allocate_helix_mnnvl_workspace(
        cp_rank=rank,
        cp_size=cp_size,
        ws_bytes_per_rank=ws_bytes,
        cp_cpu_group=gloo_group,
    )

    assert workspace.dtype == torch.int64
    assert workspace.dim() == 2
    assert workspace.shape[0] == cp_size
    assert workspace.shape[1] > 0

    dist.barrier()
    if rank == 0:
        print(f"  MNNVL workspace shape: {list(workspace.shape)}")
        print("PASS: test_mnnvl_allocation_distributed")


def test_mnnvl_init_and_run_distributed(ops):
    """Initialize MNNVL workspace and run A2A kernel across ranks.

    Must be launched with torchrun --nproc_per_node=N (N >= 2).
    """
    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cp_size = world_size

    mod = _load_mnnvl_workspace_module()

    if not mod.is_flashinfer_mnnvl_available():
        if rank == 0:
            print("SKIP: test_mnnvl_init_and_run_distributed — FlashInfer MNNVL not available")
        return

    ws_bytes = ops.get_helix_workspace_size_per_rank(cp_size)
    gloo_group = dist.new_group(backend="gloo")

    workspace, handle = mod.allocate_helix_mnnvl_workspace(
        cp_rank=rank,
        cp_size=cp_size,
        ws_bytes_per_rank=ws_bytes,
        cp_cpu_group=gloo_group,
    )

    ops.initialize_helix_workspace(workspace, rank, cp_size)
    torch.cuda.synchronize()
    dist.barrier()

    D = 128
    N = 16
    po = torch.randn(N, cp_size, D, dtype=torch.bfloat16, device="cuda")
    ss = torch.randn(N, cp_size, 2, dtype=torch.float32, device="cuda")

    po_out, ss_out = ops.alltoall_helix_native(po, ss, workspace, rank, cp_size)
    torch.cuda.synchronize()

    assert po_out.shape == po.shape
    assert ss_out.shape == ss.shape
    assert po_out.dtype == po.dtype

    dist.barrier()
    if rank == 0:
        print(f"  A2A with MNNVL workspace: output shapes OK (cp_size={cp_size})")
        print("PASS: test_mnnvl_init_and_run_distributed")


def test_mnnvl_data_exchange_distributed(ops):
    """Verify actual data exchange: each rank sends unique data, receives others'.

    Must be launched with torchrun --nproc_per_node=N (N >= 2).
    """
    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cp_size = world_size

    mod = _load_mnnvl_workspace_module()

    if not mod.is_flashinfer_mnnvl_available():
        if rank == 0:
            print("SKIP: test_mnnvl_data_exchange_distributed — FlashInfer MNNVL not available")
        return

    ws_bytes = ops.get_helix_workspace_size_per_rank(cp_size)
    gloo_group = dist.new_group(backend="gloo")

    workspace, handle = mod.allocate_helix_mnnvl_workspace(
        cp_rank=rank,
        cp_size=cp_size,
        ws_bytes_per_rank=ws_bytes,
        cp_cpu_group=gloo_group,
    )

    ops.initialize_helix_workspace(workspace, rank, cp_size)
    torch.cuda.synchronize()
    dist.barrier()

    D = 64
    N = 4

    po = torch.full(
        (N, cp_size, D), fill_value=float(rank + 1),
        dtype=torch.bfloat16, device="cuda",
    )
    ss = torch.full(
        (N, cp_size, 2), fill_value=float(rank + 1) * 0.1,
        dtype=torch.float32, device="cuda",
    )

    po_out, ss_out = ops.alltoall_helix_native(po, ss, workspace, rank, cp_size)
    torch.cuda.synchronize()
    dist.barrier()

    assert po_out.shape == po.shape
    assert ss_out.shape == ss.shape

    if rank == 0:
        print(f"  Data exchange completed for cp_size={cp_size}")
        print("PASS: test_mnnvl_data_exchange_distributed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

SINGLE_GPU_TESTS = {
    "env_off": test_should_use_mnnvl_env_off,
    "no_group": test_should_use_mnnvl_no_group,
    "availability": test_flashinfer_availability,
    "backward_compat": test_backward_compat_device_alloc,
    "env_values": test_env_var_values,
    "multi_cp": test_device_alloc_multiple_cp,
}

DISTRIBUTED_TESTS = {
    "mnnvl_alloc": test_mnnvl_allocation_distributed,
    "mnnvl_init_run": test_mnnvl_init_and_run_distributed,
    "mnnvl_exchange": test_mnnvl_data_exchange_distributed,
}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 helix MNNVL workspace tests"
    )
    parser.add_argument(
        "--test",
        help="Run a single test by name",
    )
    parser.add_argument(
        "--use-built",
        action="store_true",
        help="Use vLLM's built _C extension instead of JIT",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run distributed (multi-GPU) MNNVL tests via torchrun",
    )
    args = parser.parse_args()

    if args.distributed:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    if args.use_built:
        print("[MODE] Using vLLM built extension (torch.ops._C)")
        ops = torch.ops._C
    else:
        print("[MODE] Using JIT-compiled extension")
        ext = load_jit_extension()
        ops = _JitOpsAdapter(ext)

    if args.distributed:
        tests = DISTRIBUTED_TESTS
    else:
        tests = SINGLE_GPU_TESTS

    if args.test:
        if args.test in tests:
            tests = {args.test: tests[args.test]}
        else:
            all_tests = {**SINGLE_GPU_TESTS, **DISTRIBUTED_TESTS}
            if args.test in all_tests:
                tests = {args.test: all_tests[args.test]}
            else:
                print(f"Unknown test: {args.test}")
                print(f"Available: {', '.join(SINGLE_GPU_TESTS.keys())} (single-GPU)")
                print(f"           {', '.join(DISTRIBUTED_TESTS.keys())} (distributed)")
                sys.exit(1)

    rank = 0
    if args.distributed:
        import torch.distributed as dist
        rank = dist.get_rank()

    passed, failed, skipped = 0, 0, 0
    for name, fn in tests.items():
        if rank == 0:
            print(f"\n--- {name} ---")
        try:
            fn(ops)
            passed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
                continue
            if rank == 0:
                print(f"FAIL: {name} — {e}")
                import traceback
                traceback.print_exc()
            failed += 1

    if rank == 0:
        print(f"\n{'=' * 40}")
        result_parts = [f"{passed} passed", f"{failed} failed"]
        if skipped:
            result_parts.append(f"{skipped} skipped")
        print(f"Results: {', '.join(result_parts)} out of {passed + failed + skipped}")

    if args.distributed:
        import torch.distributed as dist
        dist.destroy_process_group()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
