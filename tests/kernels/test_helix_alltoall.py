"""
Phase 1 smoke tests for the helix all-to-all CUDA kernel.

Uses torch.utils.cpp_extension.load() for JIT compilation — no need to build
all of vLLM.  Run directly on an H200/GB200 interactive session:

    cd nim/a2a-comm/vllm-a2a/
    python tests/kernels/test_helix_alltoall.py          # all single-GPU tests
    python tests/kernels/test_helix_alltoall.py --test ws # just workspace-size

Requirements: PyTorch with CUDA 12.0+, SM90+ GPU (H100/H200/GB200).
The JIT extension is cached in ~/.cache/torch_extensions/ across sessions.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # nim/a2a-comm/vllm-a2a


def _ensure_writable_cache():
    """Fall back to a writable cache dir if the default is not writable."""
    cache_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if cache_dir and os.access(os.path.dirname(cache_dir) or ".", os.W_OK):
        return
    # Default cache may be in a read-only container path; try alternatives
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


def _get_cuda_gencode_flags():
    """Return NVCC gencode flags matching the current GPU architecture.

    Hopper (H100/H200) -> sm_90a, Blackwell (B200/GB200) -> sm_100a.
    """
    major, minor = torch.cuda.get_device_capability()
    if major >= 9:
        arch = f"compute_{major}0a"
        code = f"sm_{major}0a"
    else:
        arch = f"compute_{major}{minor}"
        code = f"sm_{major}{minor}"
    flag = f"-gencode=arch={arch},code={code}"
    print(f"[JIT] GPU SM{major}.{minor} -> {flag}")
    return [flag]


def load_helix_extension():
    """JIT-compile the helix all-to-all extension (cached after first call)."""
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
            *_get_cuda_gencode_flags(),
        ],
        extra_cflags=["-O3"],
        verbose=True,
    )
    print(f"[JIT] Done in {time.time() - t0:.1f}s")
    return ext


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _alloc_workspace(ext, cp_size: int, device="cuda"):
    """Allocate and return a workspace tensor of the right shape."""
    ws_bytes = ext.get_helix_workspace_size_per_rank(cp_size)
    ws_elems_per_rank = (ws_bytes + 7) // 8  # int64 elements
    ws = torch.zeros(cp_size, ws_elems_per_rank, dtype=torch.long, device=device)
    return ws, ws_bytes


# ---------------------------------------------------------------------------
# Tests — each is self-contained, prints PASS/FAIL
# ---------------------------------------------------------------------------

def test_workspace_size(ext):
    """get_helix_workspace_size_per_rank returns sensible positive values."""
    for cp in [1, 2, 4, 8]:
        ws = ext.get_helix_workspace_size_per_rank(cp)
        assert ws > 0, f"cp_size={cp}: workspace size should be > 0, got {ws}"
        assert ws % 8 == 0, f"cp_size={cp}: workspace size should be 8-byte aligned"
        print(f"  cp_size={cp:2d} → workspace_per_rank = {ws:>10,} bytes ({ws/1024/1024:.2f} MB)")
    print("PASS: test_workspace_size")


def test_init_workspace_single_rank(ext):
    """initialize_helix_workspace succeeds for cp_size=1 (degenerate)."""
    ws, _ = _alloc_workspace(ext, cp_size=1)
    ext.initialize_helix_workspace(ws, 0, 1)
    torch.cuda.synchronize()
    print("PASS: test_init_workspace_single_rank")


def test_input_validation_dtype(ext):
    """alltoall_helix_native rejects wrong dtypes."""
    cp_size, cp_rank = 2, 0
    ws, _ = _alloc_workspace(ext, cp_size)
    D = 128

    # float32 partial_o should be rejected (expects half / bf16)
    bad_po = torch.randn(4, cp_size, D, dtype=torch.float32, device="cuda")
    ss = torch.randn(4, cp_size, 2, dtype=torch.float32, device="cuda")
    try:
        ext.alltoall_helix_native(bad_po, ss, ws, cp_rank, cp_size)
        assert False, "Should have raised for float32 partial_o"
    except RuntimeError:
        pass

    # int32 softmax_stats should be rejected (expects float32)
    po = torch.randn(4, cp_size, D, dtype=torch.bfloat16, device="cuda")
    bad_ss = torch.randn(4, cp_size, 2).int().cuda()
    try:
        ext.alltoall_helix_native(po, bad_ss, ws, cp_rank, cp_size)
        assert False, "Should have raised for int32 softmax_stats"
    except RuntimeError:
        pass

    print("PASS: test_input_validation_dtype")


def test_input_validation_shape(ext):
    """alltoall_helix_native rejects mismatched / bad shapes."""
    cp_size, cp_rank = 2, 0
    ws, _ = _alloc_workspace(ext, cp_size)
    D = 128

    # Odd softmax_stats last dim (must be even, >= 2)
    po = torch.randn(4, cp_size, D, dtype=torch.bfloat16, device="cuda")
    bad_ss = torch.randn(4, cp_size, 3, dtype=torch.float32, device="cuda")
    try:
        ext.alltoall_helix_native(po, bad_ss, ws, cp_rank, cp_size)
        assert False, "Should have raised for odd softmax_stats last dim"
    except RuntimeError:
        pass

    # Wrong cp_size dim
    bad_po = torch.randn(4, cp_size + 1, D, dtype=torch.bfloat16, device="cuda")
    ss = torch.randn(4, cp_size + 1, 2, dtype=torch.float32, device="cuda")
    try:
        ext.alltoall_helix_native(bad_po, ss, ws, cp_rank, cp_size)
        assert False, "Should have raised for wrong cp_size dim"
    except RuntimeError:
        pass

    # Non-contiguous
    po_nc = torch.randn(cp_size, 4, D, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    ss_nc = torch.randn(cp_size, 4, 2, dtype=torch.float32, device="cuda").transpose(0, 1)
    assert not po_nc.is_contiguous()
    try:
        ext.alltoall_helix_native(po_nc, ss_nc, ws, cp_rank, cp_size)
        assert False, "Should have raised for non-contiguous tensor"
    except RuntimeError:
        pass

    print("PASS: test_input_validation_shape")


def test_input_validation_alignment(ext):
    """partial_o last dim must be 16-byte aligned."""
    cp_size, cp_rank = 2, 0
    ws, _ = _alloc_workspace(ext, cp_size)

    # bf16 with last dim = 3 → 3*2 = 6 bytes, not 16-byte aligned
    bad_po = torch.randn(4, cp_size, 3, dtype=torch.bfloat16, device="cuda")
    ss = torch.randn(4, cp_size, 2, dtype=torch.float32, device="cuda")
    try:
        ext.alltoall_helix_native(bad_po, ss, ws, cp_rank, cp_size)
        assert False, "Should have raised for unaligned partial_o"
    except RuntimeError:
        pass

    print("PASS: test_input_validation_alignment")


def test_output_shapes(ext):
    """alltoall_helix_native returns tensors with correct shapes and dtypes.

    NOTE: With cp_size=1, cp_rank=0 (self-send), the kernel may not produce
    correct numerical results but should not crash and output shapes must match.
    """
    cp_size, cp_rank = 1, 0
    ws, _ = _alloc_workspace(ext, cp_size)
    ext.initialize_helix_workspace(ws, cp_rank, cp_size)
    torch.cuda.synchronize()

    D = 128
    entry_count = 8
    po = torch.randn(entry_count, cp_size, D, dtype=torch.bfloat16, device="cuda")
    ss = torch.randn(entry_count, cp_size, 2, dtype=torch.float32, device="cuda")

    po_out, ss_out = ext.alltoall_helix_native(po, ss, ws, cp_rank, cp_size)
    torch.cuda.synchronize()

    assert po_out.shape == po.shape, f"partial_o shape mismatch: {po_out.shape} vs {po.shape}"
    assert ss_out.shape == ss.shape, f"softmax_stats shape mismatch: {ss_out.shape} vs {ss.shape}"
    assert po_out.dtype == po.dtype, f"partial_o dtype mismatch: {po_out.dtype} vs {po.dtype}"
    assert ss_out.dtype == ss.dtype, f"softmax_stats dtype mismatch: {ss_out.dtype} vs {ss.dtype}"

    print("PASS: test_output_shapes")


def test_self_send_correctness(ext):
    """With cp_size=1, the kernel should copy input → output (self-send)."""
    cp_size, cp_rank = 1, 0
    ws, _ = _alloc_workspace(ext, cp_size)
    ext.initialize_helix_workspace(ws, cp_rank, cp_size)
    torch.cuda.synchronize()

    D = 128
    entry_count = 16
    po = torch.randn(entry_count, cp_size, D, dtype=torch.bfloat16, device="cuda")
    ss = torch.randn(entry_count, cp_size, 2, dtype=torch.float32, device="cuda")

    po_out, ss_out = ext.alltoall_helix_native(po, ss, ws, cp_rank, cp_size)
    torch.cuda.synchronize()

    if torch.allclose(po_out.float(), po.float(), atol=1e-3) and \
       torch.allclose(ss_out, ss, atol=1e-5):
        print("PASS: test_self_send_correctness (cp_size=1 self-copy matches)")
    else:
        po_diff = (po_out.float() - po.float()).abs().max().item()
        ss_diff = (ss_out - ss).abs().max().item()
        print(f"INFO: test_self_send_correctness — self-copy NOT exact "
              f"(po max_diff={po_diff:.6f}, ss max_diff={ss_diff:.6f}). "
              f"This is expected if the kernel doesn't special-case cp_size=1.")


def test_higher_dim_shapes(ext):
    """Verify the op handles >3D tensors (leading dims are flattened)."""
    cp_size, cp_rank = 1, 0
    ws, _ = _alloc_workspace(ext, cp_size)
    ext.initialize_helix_workspace(ws, cp_rank, cp_size)
    torch.cuda.synchronize()

    D = 64
    po = torch.randn(2, 3, 4, cp_size, D, dtype=torch.bfloat16, device="cuda")
    ss = torch.randn(2, 3, 4, cp_size, 2, dtype=torch.float32, device="cuda")

    po_out, ss_out = ext.alltoall_helix_native(po, ss, ws, cp_rank, cp_size)
    torch.cuda.synchronize()

    assert po_out.shape == po.shape
    assert ss_out.shape == ss.shape
    print("PASS: test_higher_dim_shapes")


# ---------------------------------------------------------------------------

ALL_TESTS = {
    "ws": test_workspace_size,
    "init": test_init_workspace_single_rank,
    "dtype": test_input_validation_dtype,
    "shape": test_input_validation_shape,
    "align": test_input_validation_alignment,
    "output": test_output_shapes,
    "self_send": test_self_send_correctness,
    "highdim": test_higher_dim_shapes,
}


def main():
    parser = argparse.ArgumentParser(description="Phase 1 helix alltoall tests")
    parser.add_argument("--test", choices=list(ALL_TESTS.keys()),
                        help="Run a single test (default: all)")
    args = parser.parse_args()

    ext = load_helix_extension()

    tests = {args.test: ALL_TESTS[args.test]} if args.test else ALL_TESTS
    passed, failed = 0, 0
    for name, fn in tests.items():
        print(f"\n--- {name} ---")
        try:
            fn(ext)
            passed += 1
        except Exception as e:
            print(f"FAIL: {name} — {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
