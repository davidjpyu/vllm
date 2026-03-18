# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the Helix A2A integration with FlashInfer.

Test categories:
  1. Config validation (no GPU needed)
  2. Reshape logic (GPU, no multi-rank)
  3. LSE combine correctness (GPU, no multi-rank)
  4. FlashInfer workspace lifecycle (SM90+ GPU, single-GPU multi-rank)
  5. Full A2A + combine (SM90+ GPU, single-GPU multi-rank)

Tests 4-5 require FlashInfer with Helix support and SM90+ GPUs.

Run:
    python -m pytest tests/v1/attention/test_helix_flashinfer.py -v -s
"""

import pytest
import torch

# ─── SM90+ / FlashInfer availability gates ───────────────────────────────


def _sm90_available() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 9
    except Exception:
        return False


def _flashinfer_helix_available() -> bool:
    try:
        from flashinfer.comm import (
            helix_a2a_alltoall,
            helix_a2a_allocate_workspace,
            helix_a2a_init_workspace,
            helix_a2a_workspace_size,
        )
        return _sm90_available()
    except ImportError:
        return False


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA GPU",
)
requires_sm90 = pytest.mark.skipif(
    not _sm90_available(),
    reason="Requires SM90+ GPU (Hopper/Blackwell)",
)
requires_flashinfer_helix = pytest.mark.skipif(
    not _flashinfer_helix_available(),
    reason="Requires FlashInfer with Helix A2A support and SM90+ GPU",
)


# ============================================================================
# 1. Config validation tests (no GPU needed)
# ============================================================================


class TestHelixConfig:
    """Verify helix_mode and helix_a2a_backend config validation."""

    def test_helix_mode_requires_dcp_gt_1(self):
        from vllm.config.parallel import ParallelConfig
        with pytest.raises(ValueError, match="decode_context_parallel_size > 1"):
            ParallelConfig(
                tensor_parallel_size=4,
                decode_context_parallel_size=1,
                helix_mode=True,
            )

    def test_flashinfer_native_requires_helix_mode(self):
        from vllm.config.parallel import ParallelConfig
        with pytest.raises(ValueError, match="requires helix_mode=True"):
            ParallelConfig(
                tensor_parallel_size=4,
                decode_context_parallel_size=4,
                helix_mode=False,
                helix_a2a_backend="flashinfer_native",
            )

    def test_valid_helix_config(self):
        from vllm.config.parallel import ParallelConfig
        cfg = ParallelConfig(
            tensor_parallel_size=4,
            decode_context_parallel_size=4,
            helix_mode=True,
            helix_a2a_backend="flashinfer_native",
        )
        assert cfg.helix_mode is True
        assert cfg.helix_a2a_backend == "flashinfer_native"
        assert cfg.helix_kvp_size == 4
        assert cfg.helix_tpa_size == 1

    def test_helix_kvp_tpa_properties(self):
        from vllm.config.parallel import ParallelConfig
        cfg = ParallelConfig(
            tensor_parallel_size=4,
            decode_context_parallel_size=2,
            helix_mode=True,
        )
        assert cfg.helix_kvp_size == 2
        assert cfg.helix_tpa_size == 2

    def test_helix_disabled_properties(self):
        from vllm.config.parallel import ParallelConfig
        cfg = ParallelConfig(
            tensor_parallel_size=4,
            decode_context_parallel_size=1,
            helix_mode=False,
        )
        assert cfg.helix_kvp_size == 1
        assert cfg.helix_tpa_size == 4


# ============================================================================
# 2. Reshape logic tests (GPU, no multi-rank comms)
# ============================================================================


@requires_cuda
class TestReshapeLogic:
    """Verify the tensor reshape contracts match the FlashInfer kernel spec."""

    @pytest.mark.parametrize("B,N,H_per_rank,D", [
        (1, 2, 4, 128),
        (16, 4, 8, 128),
        (32, 2, 16, 256),
    ])
    def test_output_reshape_roundtrip(self, B, N, H_per_rank, D):
        """Verify [B,H,D] → [E,N,D] → [N,B,H/N,D] roundtrip."""
        H = N * H_per_rank
        local_output = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda")

        entry_count = B * H_per_rank
        partial_o = (
            local_output.view(B, N, H_per_rank, D)
            .permute(0, 2, 1, 3)
            .reshape(entry_count, N, D)
            .contiguous()
        )

        assert partial_o.shape == (entry_count, N, D)

        # Simulate identity all-to-all (no exchange)
        partial_o_out = partial_o.clone()

        # Reshape back
        recv_output = (
            partial_o_out.view(B, H_per_rank, N, D)
            .permute(2, 0, 1, 3)
            .contiguous()
        )

        assert recv_output.shape == (N, B, H_per_rank, D)

        # Verify: recv_output[n, b, h, :] == local_output[b, n*H_per_rank+h, :]
        for n in range(N):
            expected = local_output[:, n * H_per_rank:(n + 1) * H_per_rank, :]
            torch.testing.assert_close(recv_output[n], expected, atol=0, rtol=0)

    @pytest.mark.parametrize("B,N,H_per_rank", [
        (1, 2, 4),
        (16, 4, 8),
    ])
    def test_lse_reshape_roundtrip(self, B, N, H_per_rank):
        """Verify [B,H] → [E,N,2] → [N,B,H/N] roundtrip."""
        H = N * H_per_rank
        local_lse = torch.randn(B, H, dtype=torch.float32, device="cuda")

        entry_count = B * H_per_rank
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

        assert softmax_stats.shape == (entry_count, N, 2)

        # Simulate identity all-to-all
        ss_out = softmax_stats.clone()

        recv_lse = (
            ss_out[:, :, 0]
            .view(B, H_per_rank, N)
            .permute(2, 0, 1)
            .contiguous()
        )

        assert recv_lse.shape == (N, B, H_per_rank)

        for n in range(N):
            expected = local_lse[:, n * H_per_rank:(n + 1) * H_per_rank]
            torch.testing.assert_close(recv_lse[n], expected, atol=0, rtol=0)


# ============================================================================
# 3. LSE combine correctness tests (GPU, no multi-rank comms)
# ============================================================================


@requires_cuda
class TestLSECombine:
    """Verify Triton LSE combine matches the CPU reference."""

    @pytest.mark.parametrize("N,B,H,D", [
        (2, 4, 8, 128),
        (4, 16, 16, 128),
        (2, 1, 4, 256),
    ])
    @pytest.mark.parametrize("is_base_e", [True, False])
    def test_triton_matches_reference(self, N, B, H, D, is_base_e):
        from vllm.v1.attention.ops.helix import (
            _lse_weighted_combine,
            helix_lse_combine_triton,
        )

        recv_output = torch.randn(
            N, B, H, D, dtype=torch.bfloat16, device="cuda",
        )
        recv_lse = torch.randn(
            N, B, H, dtype=torch.float32, device="cuda",
        )

        ref_out, ref_lse = _lse_weighted_combine(
            recv_output, recv_lse, return_lse=True, is_lse_base_on_e=is_base_e,
        )
        triton_out, triton_lse = helix_lse_combine_triton(
            recv_output, recv_lse, return_lse=True, is_lse_base_on_e=is_base_e,
        )

        torch.testing.assert_close(
            triton_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            triton_lse, ref_lse, atol=1e-2, rtol=1e-2,
        )


# ============================================================================
# 4. FlashInfer workspace lifecycle tests (SM90+ GPU)
# ============================================================================


@requires_flashinfer_helix
class TestFlashInferWorkspace:
    """Verify FlashInfer workspace allocation and initialization."""

    def test_workspace_size_positive(self):
        from flashinfer.comm import helix_a2a_workspace_size
        for cp_size in [2, 4]:
            ws = helix_a2a_workspace_size(cp_size)
            assert isinstance(ws, int)
            assert ws > 0

    def test_allocate_workspace_shape(self):
        from flashinfer.comm import (
            helix_a2a_allocate_workspace,
            helix_a2a_workspace_size,
        )
        for cp_size in [2, 4]:
            ws_bytes = helix_a2a_workspace_size(cp_size)
            workspace = helix_a2a_allocate_workspace(cp_size, cp_rank=0)
            assert workspace.dtype == torch.int64
            assert workspace.shape[0] == cp_size
            assert workspace.shape[1] == (ws_bytes + 7) // 8

    def test_init_workspace_does_not_hang(self):
        from flashinfer.comm import (
            helix_a2a_allocate_workspace,
            helix_a2a_init_workspace,
        )
        for cp_size in [2, 4]:
            workspace = helix_a2a_allocate_workspace(cp_size, cp_rank=0)
            for r in range(cp_size):
                helix_a2a_init_workspace(workspace, r, cp_size)
            torch.cuda.synchronize()


# ============================================================================
# 5. Full A2A + combine (SM90+, single-GPU multi-rank simulation)
# ============================================================================


def _to_torch(t):
    """Convert tvm_ffi.core.Tensor or DLPack object to torch.Tensor."""
    if isinstance(t, torch.Tensor):
        return t
    return torch.from_dlpack(t)


@requires_flashinfer_helix
class TestFlashInferA2ACorrectness:
    """End-to-end correctness: FlashInfer A2A transpose property."""

    @pytest.mark.parametrize("cp_size,B,D,S,dtype", [
        (2, 16, 128, 2, torch.bfloat16),
        (4, 16, 128, 2, torch.bfloat16),
        (2, 16, 128, 2, torch.float16),
    ])
    def test_alltoall_transpose(self, cp_size, B, D, S, dtype):
        """recv[r][.., peer, :] == input[peer][.., r, :]."""
        from flashinfer.comm import (
            helix_a2a_alltoall,
            helix_a2a_allocate_workspace,
            helix_a2a_init_workspace,
        )

        torch.cuda.set_device(0)
        workspace = helix_a2a_allocate_workspace(cp_size, cp_rank=0)

        all_po = [
            torch.randn(B, cp_size, D, dtype=dtype, device="cuda")
            for _ in range(cp_size)
        ]
        all_ss = [
            torch.randn(B, cp_size, S, dtype=torch.float32, device="cuda")
            for _ in range(cp_size)
        ]

        for r in range(cp_size):
            helix_a2a_init_workspace(workspace, r, cp_size)
        torch.cuda.synchronize()

        streams = [torch.cuda.Stream() for _ in range(cp_size)]
        recv_o = [None] * cp_size
        recv_s = [None] * cp_size

        for r in range(cp_size):
            with torch.cuda.stream(streams[r]):
                o, s = helix_a2a_alltoall(
                    all_po[r], all_ss[r], workspace, r, cp_size,
                )
                recv_o[r] = _to_torch(o)
                recv_s[r] = _to_torch(s)

        for stream in streams:
            stream.synchronize()
        torch.cuda.synchronize()

        for r in range(cp_size):
            for peer in range(cp_size):
                torch.testing.assert_close(
                    recv_o[r][..., peer, :],
                    all_po[peer][..., r, :],
                    atol=0, rtol=0,
                )
                torch.testing.assert_close(
                    recv_s[r][..., peer, :],
                    all_ss[peer][..., r, :],
                    atol=0, rtol=0,
                )


@requires_flashinfer_helix
class TestHelixAllToAllFlashInferManager:
    """Test the HelixAllToAllFlashInfer workspace manager."""

    def setup_method(self):
        from vllm.distributed.helix_alltoall_flashinfer import (
            HelixAllToAllFlashInfer,
        )
        HelixAllToAllFlashInfer.clear_cache()

    def test_manager_caches_by_key(self):
        """Verify workspace is cached per (cp_rank, cp_size)."""
        from vllm.distributed.helix_alltoall_flashinfer import (
            HelixAllToAllFlashInfer,
        )

        # Use direct allocation (no process group)
        mgr1 = HelixAllToAllFlashInfer.get(cp_rank=0, cp_size=2)
        mgr2 = HelixAllToAllFlashInfer.get(cp_rank=0, cp_size=2)
        assert mgr1 is mgr2

    def test_manager_run_returns_correct_shapes(self):
        """Verify run() returns tensors of the correct shape."""
        from vllm.distributed.helix_alltoall_flashinfer import (
            HelixAllToAllFlashInfer,
        )

        cp_size = 2
        B, D, S = 16, 128, 2

        # Initialize for all simulated ranks
        for r in range(cp_size):
            mgr = HelixAllToAllFlashInfer.get(cp_rank=r, cp_size=cp_size)

        mgr = HelixAllToAllFlashInfer.get(cp_rank=0, cp_size=cp_size)

        partial_o = torch.randn(
            B, cp_size, D, dtype=torch.bfloat16, device="cuda",
        )
        softmax_stats = torch.randn(
            B, cp_size, S, dtype=torch.float32, device="cuda",
        )

        recv_o, recv_s = mgr.run(partial_o, softmax_stats)

        assert recv_o.shape == partial_o.shape
        assert recv_s.shape == softmax_stats.shape
        assert recv_o.dtype == partial_o.dtype
        assert recv_s.dtype == softmax_stats.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
