# DFlash Pending Work

> Last updated: 2026-03-21

## Completed

- [x] Resume Phase C training — resumed from step 17,001
- [x] Inference benchmark — τ = 1.86 at step 18,001 (below target 3.0)
- [x] SpecForge cross-check — identified 2 training quality bugs

## Active — Bug Fixes

- [ ] **Fix Bug 1: Zero-loss dummy on empty anchors** (`dflash.py:128-134`)
  Replace silent zero-loss return with `raise ValueError` or proper fallback (uniform stride anchors on valid positions). See [Issues](dflash_issues.md#bug-1-zero-loss-dummy-on-empty-anchors-dflashpy128-134).

- [ ] **Investigate Bug 2: Anchor filtering by anchor mask** (`dflash.py:126`)
  Benchmark whether allowing anchors at prompt positions (where labels are in completion region) improves τ. See [Issues](dflash_issues.md#bug-2-anchor-filtering-by-anchor-positions-mask-dflashpy126).

- [ ] **Retrain with fixes**: Apply bug fixes and retrain from scratch or resume, then re-benchmark τ. Target: τ ≥ 3.0.

## Active — Code Improvements

- [ ] **Commit factory.py timeout fix**: Increase `find_free_port` timeout from 30s to 120s for PyTorch 2.9+ compatibility (currently local pod patch only).

- [ ] **Draft KV cache**: Add KV cache support to `DFlashDraftModel.forward()` (currently recomputes full context each cycle — O(n²) scaling).

## Future

- [ ] **Eagle3 inference comparison**: Side-by-side benchmark of DFlash vs Eagle3.
- [ ] **PyTorch speed regression**: Profile with `torch.profiler` to identify exact kernel-level regression in 2.9.1, or test torch 2.7/2.8.
- [ ] **Port remaining SpecForge improvements**: Check for additional commits after `507da3e`.
