import os
import unittest

import torch
import torch.nn.functional as F


def _old_forward_kl(logits: torch.Tensor, target_p: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=-1)
    return -(target_p * log_p).sum(-1)


def _new_forward_kl(logits: torch.Tensor, target_p: torch.Tensor) -> torch.Tensor:
    logits_f32 = logits.float()
    return torch.logsumexp(logits_f32, dim=-1) - (target_p * logits_f32).sum(-1)


def _old_softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits.float(), dim=-1)


def _new_softmax(logits: torch.Tensor) -> torch.Tensor:
    logits_f32 = logits.float()
    return torch.exp(logits_f32 - torch.logsumexp(logits_f32, dim=-1, keepdim=True))


def _bench_cuda(fn, *args, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        out = fn(*args)
        if torch.is_tensor(out):
            out.sum().backward() if out.requires_grad else None
        for arg in args:
            if torch.is_tensor(arg) and arg.grad is not None:
                arg.grad = None
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn(*args)
        if torch.is_tensor(out):
            out.sum().backward() if out.requires_grad else None
        for arg in args:
            if torch.is_tensor(arg) and arg.grad is not None:
                arg.grad = None
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@unittest.skipUnless(
    os.environ.get("TORCHSPEC_RUN_PERF_TESTS") == "1",
    "set TORCHSPEC_RUN_PERF_TESTS=1 to run CUDA perf comparisons",
)
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this perf comparison")
class TestLossSoftmaxPerf(unittest.TestCase):
    def test_forward_kl_formula_perf(self):
        torch.manual_seed(0)
        device = "cuda"
        n_tokens = int(os.environ.get("TORCHSPEC_PERF_TOKENS", "512"))
        vocab_size = int(os.environ.get("TORCHSPEC_PERF_VOCAB", "32000"))
        dtype = torch.bfloat16

        logits_old = torch.randn(n_tokens, vocab_size, device=device, dtype=dtype).requires_grad_()
        logits_new = logits_old.detach().clone().requires_grad_()
        target_p = F.softmax(torch.randn(n_tokens, vocab_size, device=device).float(), dim=-1)

        torch.testing.assert_close(
            _new_forward_kl(logits_new, target_p),
            _old_forward_kl(logits_old, target_p),
            atol=1e-4,
            rtol=1e-4,
        )

        old_compiled = torch.compile(_old_forward_kl, dynamic=None)
        new_compiled = torch.compile(_new_forward_kl, dynamic=None)
        old_ms = _bench_cuda(old_compiled, logits_old, target_p)
        new_ms = _bench_cuda(new_compiled, logits_new, target_p)
        print(
            f"forward_kl n_tokens={n_tokens} vocab={vocab_size}: "
            f"old_log_softmax={old_ms:.3f}ms new_logsumexp={new_ms:.3f}ms "
            f"ratio={new_ms / old_ms:.3f}"
        )

    def test_target_softmax_formula_perf(self):
        torch.manual_seed(0)
        device = "cuda"
        n_tokens = int(os.environ.get("TORCHSPEC_PERF_TOKENS", "512"))
        vocab_size = int(os.environ.get("TORCHSPEC_PERF_VOCAB", "32000"))
        dtype = torch.bfloat16
        logits = torch.randn(n_tokens, vocab_size, device=device, dtype=dtype)

        torch.testing.assert_close(
            _new_softmax(logits),
            _old_softmax(logits),
            atol=1e-4,
            rtol=1e-4,
        )

        old_compiled = torch.compile(_old_softmax, dynamic=None)
        new_compiled = torch.compile(_new_softmax, dynamic=None)
        old_ms = _bench_cuda(old_compiled, logits)
        new_ms = _bench_cuda(new_compiled, logits)
        print(
            f"target_softmax n_tokens={n_tokens} vocab={vocab_size}: "
            f"old_softmax={old_ms:.3f}ms new_exp_logsumexp={new_ms:.3f}ms "
            f"ratio={new_ms / old_ms:.3f}"
        )
