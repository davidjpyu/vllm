"""
Phase 4 standalone validation: backend dispatch fallback and reshape logic.

No multi-GPU or vLLM install required — runs on a single GPU.

    cd /home/scratch.davidyu_coreai/workspace/helix/vllm-a2a
    python tests/kernels/test_helix_phase4_reshape.py

Requirements: PyTorch with CUDA, SM90+ GPU.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch


def test_backend_fallback():
    """_get_helix_a2a_backend() should return 'nccl' when vLLM config is not
    initialized (no running engine)."""
    from vllm.v1.attention.ops.helix import _get_helix_a2a_backend

    backend = _get_helix_a2a_backend()
    assert backend == "nccl", f"Expected 'nccl' fallback, got '{backend}'"
    print(f"PASS: _get_helix_a2a_backend() fallback = '{backend}'")


def test_input_reshape():
    """Verify the reshape from helix_alltoall_lse_reduce inputs into the
    native kernel's expected [entry_count, cp_size, ...] layout."""
    B, H, D, N = 4, 8, 64, 2
    H_per_rank = H // N
    entry_count = B * H_per_rank

    local_output = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda")
    local_lse = torch.randn(B, H, dtype=torch.float32, device="cuda")

    partial_o = (
        local_output.view(B, N, H_per_rank, D)
        .permute(0, 2, 1, 3)
        .reshape(entry_count, N, D)
        .contiguous()
    )
    assert partial_o.shape == (entry_count, N, D), \
        f"Bad partial_o: {partial_o.shape}"

    lse_permuted = (
        local_lse.view(B, N, H_per_rank)
        .permute(0, 2, 1)
        .reshape(entry_count, N)
        .contiguous()
    )
    softmax_stats = torch.zeros(
        entry_count, N, 2, dtype=torch.float32, device="cuda",
    )
    softmax_stats[:, :, 0] = lse_permuted
    assert softmax_stats.shape == (entry_count, N, 2), \
        f"Bad softmax_stats: {softmax_stats.shape}"

    print(f"PASS: input reshape — partial_o {partial_o.shape}, "
          f"softmax_stats {softmax_stats.shape}")


def test_output_reshape():
    """Verify the reshape from native kernel outputs back to the
    [N, B, H/N, D] / [N, B, H/N] layout expected by helix_lse_combine_triton."""
    B, H, D, N = 4, 8, 64, 2
    H_per_rank = H // N
    entry_count = B * H_per_rank

    partial_o_out = torch.randn(
        entry_count, N, D, dtype=torch.bfloat16, device="cuda",
    )
    ss_out = torch.randn(
        entry_count, N, 2, dtype=torch.float32, device="cuda",
    )

    recv_output = (
        partial_o_out.view(B, H_per_rank, N, D)
        .permute(2, 0, 1, 3)
        .contiguous()
    )
    assert recv_output.shape == (N, B, H_per_rank, D), \
        f"Bad recv_output: {recv_output.shape}"

    recv_lse = (
        ss_out[:, :, 0]
        .view(B, H_per_rank, N)
        .permute(2, 0, 1)
        .contiguous()
    )
    assert recv_lse.shape == (N, B, H_per_rank), \
        f"Bad recv_lse: {recv_lse.shape}"

    print(f"PASS: output reshape — recv_output {recv_output.shape}, "
          f"recv_lse {recv_lse.shape}")


def test_roundtrip_data_integrity():
    """Verify that reshape-in followed by reshape-out preserves the data
    (no transposition bugs)."""
    B, H, D, N = 2, 4, 32, 2
    H_per_rank = H // N
    entry_count = B * H_per_rank

    local_output = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda")
    local_lse = torch.randn(B, H, dtype=torch.float32, device="cuda")

    partial_o = (
        local_output.view(B, N, H_per_rank, D)
        .permute(0, 2, 1, 3)
        .reshape(entry_count, N, D)
        .contiguous()
    )

    recv_output = (
        partial_o.view(B, H_per_rank, N, D)
        .permute(2, 0, 1, 3)
        .contiguous()
    )
    reconstructed = (
        recv_output.permute(1, 0, 2, 3)
        .reshape(B, H, D)
    )
    assert torch.allclose(reconstructed, local_output), \
        "Roundtrip mismatch for output tensor"

    lse_permuted = (
        local_lse.view(B, N, H_per_rank)
        .permute(0, 2, 1)
        .reshape(entry_count, N)
        .contiguous()
    )
    recv_lse = (
        lse_permuted.view(B, H_per_rank, N)
        .permute(2, 0, 1)
        .contiguous()
    )
    reconstructed_lse = recv_lse.permute(1, 0, 2).reshape(B, H)
    assert torch.allclose(reconstructed_lse, local_lse), \
        "Roundtrip mismatch for LSE tensor"

    print("PASS: roundtrip data integrity — reshape-in/out is lossless")


if __name__ == "__main__":
    tests = [
        ("backend_fallback", test_backend_fallback),
        ("input_reshape", test_input_reshape),
        ("output_reshape", test_output_reshape),
        ("roundtrip_data", test_roundtrip_data_integrity),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL: {name} — {e}")
            traceback.print_exc()

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
