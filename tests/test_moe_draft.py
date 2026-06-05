# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests for the DeepSeek/Kimi/GLM MoE draft block + Expert Parallel.

Single-GPU: grouped_mm expert numerics, config loading, model assembly.
Distributed (2 GPUs, spawned): EP all-to-all parity vs single-GPU, EP-aware
grad-norm clipping, and checkpoint expert gather.
"""

import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

DT = torch.bfloat16


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tiny_deepseek_config(use_moe=True):
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    return DeepseekV3Config(
        use_moe=use_moe, hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
        n_routed_experts=8, num_experts_per_tok=2, n_group=2, topk_group=1, n_shared_experts=1,
        norm_topk_prob=True, routed_scaling_factor=1.0, hidden_act="silu", num_attention_heads=4,
        num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16,
        qk_rope_head_dim=16, v_head_dim=16, vocab_size=256, num_hidden_layers=1,
        rms_norm_eps=1e-5, rope_theta=10000.0, max_position_embeddings=512,
        target_hidden_size=128, num_aux_hidden_states=3, pad_token_id=0,
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMoEDraftSingleGPU(unittest.TestCase):
    def test_grouped_mm_matches_reference(self):
        from torchspec.models.draft.moe import DeepseekV3MoEBlock, native_moe_experts_forward

        dev = "cuda"
        block = DeepseekV3MoEBlock(_tiny_deepseek_config()).to(dev, DT)
        x = torch.randn(64, 128, device=dev, dtype=DT)
        with torch.no_grad():
            rw, se = block._route(block.gate(x), x.dtype)
            native = native_moe_experts_forward(
                x, rw, se, block.experts.gate_up_proj, block.experts.down_proj, 8
            )
            # reference: naive per-expert loop
            ref = torch.zeros(64, 128, device=dev, dtype=torch.float32)
            isz = block.experts.gate_up_proj.shape[-1] // 2
            for e in range(8):
                idx = (se == e).nonzero(as_tuple=False)
                if idx.numel() == 0:
                    continue
                tok = idx[:, 0]
                gu = x[tok].to(DT) @ block.experts.gate_up_proj[e].to(DT)
                h = F.silu(gu[:, :isz]) * gu[:, isz:]
                o = h @ block.experts.down_proj[e].to(DT)
                ref.index_add_(0, tok, (o * rw[tok, idx[:, 1]].unsqueeze(-1)).float())
        rel = ((native.float() - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)).item()
        self.assertLess(rel, 5e-2)

    def test_config_loads(self):
        from torchspec.models.draft.auto import AutoDraftModelConfig

        root = os.path.join(os.path.dirname(__file__), "..", "configs", "draft_models")
        cfg = AutoDraftModelConfig.from_file(os.path.join(root, "kimi_k25_eagle3_moe.json"))
        self.assertTrue(getattr(cfg, "use_moe", False))
        self.assertEqual(cfg.n_routed_experts, 384)
        glm = AutoDraftModelConfig.from_file(os.path.join(root, "glm4_moe_eagle3.json"))
        self.assertEqual(type(glm).__name__, "Glm4MoeConfig")
        self.assertTrue(getattr(glm, "use_moe", False))

    def test_full_models_build_and_run(self):
        from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

        from torchspec.models.draft.auto import AutoEagle3DraftModel
        from torchspec.models.draft.deepseek_eagle import Eagle3DeepseekV2ForCausalLM

        dev = "cuda"
        ds = Eagle3DeepseekV2ForCausalLM(_tiny_deepseek_config(), attention_backend="sdpa").to(dev, DT)
        self.assertEqual(type(ds.midlayer.mlp).__name__, "DeepseekV3MoEBlock")

        glm_cfg = Glm4MoeConfig(
            use_moe=True, hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
            n_routed_experts=8, num_experts_per_tok=2, n_group=2, topk_group=1, n_shared_experts=1,
            num_attention_heads=8, num_key_value_heads=2, head_dim=16, vocab_size=256,
            num_hidden_layers=1, max_position_embeddings=512, target_hidden_size=128,
            num_aux_hidden_states=3,
        )
        glm = AutoEagle3DraftModel.from_config(glm_cfg, attention_backend="sdpa", torch_dtype=DT).to(dev)
        self.assertEqual(type(glm).__name__, "Glm4MoeForCausalLMEagle3")
        self.assertEqual(type(glm.midlayer.mlp).__name__, "DeepseekV3MoEBlock")
        for model in (ds, glm):
            B, S, H = 2, 16, 128
            emb = torch.randn(B, S, H, device=dev, dtype=DT)
            hid = torch.randn(B, S, H, device=dev, dtype=DT, requires_grad=True)
            pos = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)
            out, _, _ = model.backbone(emb, hid, attention_mask=None, position_ids=pos, use_cache=False)
            self.assertEqual(out.shape, (B, S, H))
            self.assertTrue(torch.isfinite(out).all())
            out.float().pow(2).mean().backward()


def _ep_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    ep_group = dist.group.WORLD

    from torchspec.models.draft.moe import native_moe_experts_forward
    from torchspec.models.draft.moe_ep import ep_moe_experts_forward
    from torchspec.training.ep_utils import clip_grad_norm_ep, gather_ep_full_state_dict

    H, isz, E, top_k, N = 128, 64, 8, 2, 40
    nl = E // world_size

    # --- EP parity: identical full weights, distinct tokens per rank ---
    torch.manual_seed(1234)
    gate_up = (torch.randn(E, H, 2 * isz, device=dev) * 0.02).to(DT)
    down = (torch.randn(E, isz, H, device=dev) * 0.02).to(DT)
    s, e = rank * nl, (rank + 1) * nl
    torch.manual_seed(100 + rank)
    hid = torch.randn(N, H, device=dev, dtype=DT)
    rw, se = torch.randn(N, E, device=dev).sigmoid().topk(top_k, dim=-1)
    rw = rw.to(DT)
    ref = native_moe_experts_forward(hid, rw, se, gate_up, down, E)
    ep = ep_moe_experts_forward(hid, rw, se, gate_up[s:e].clone(), down[s:e].clone(), E, ep_group)
    rel = ((ep.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)).item()
    assert rel < 5e-2, f"EP parity rel {rel}"

    # --- EP grad-norm clip == reference over the distributed expert set ---
    torch.manual_seed(0)
    g_non = torch.randn(H, device=dev)
    torch.manual_seed(50 + rank)
    g_exp = torch.randn(H, device=dev)

    class _P:
        def __init__(self, g, ep):
            self.grad = g
            self._is_ep = ep

    total = clip_grad_norm_ep([_P(g_non.clone(), False), _P(g_exp.clone(), True)], 1e9, ep_group=ep_group)
    gathered = [torch.zeros_like(g_exp) for _ in range(world_size)]
    dist.all_gather(gathered, g_exp)
    ref_norm = (g_non.float().pow(2).sum() + sum(t.float().pow(2).sum() for t in gathered)).sqrt()
    assert (total - ref_norm).abs().item() / ref_norm.item() < 1e-4, "EP grad norm mismatch"

    # --- ckpt gather: local experts -> full [E, ...] ---
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    from torchspec.models.draft.deepseek_eagle import Eagle3DeepseekV2ForCausalLM

    cfg = _tiny_deepseek_config()
    assert isinstance(cfg, DeepseekV3Config)
    cfg.ep_group = ep_group
    model = Eagle3DeepseekV2ForCausalLM(cfg, attention_backend="sdpa").to(dev, DT)
    full = gather_ep_full_state_dict(model, ep_group)
    gk = "midlayer.mlp.experts.gate_up_proj"
    assert full[gk].shape[0] == E
    local = model.midlayer.mlp.experts.gate_up_proj.detach()
    assert torch.equal(full[gk][rank * nl : (rank + 1) * nl].to(local.dtype), local)

    dist.destroy_process_group()


def _ep_e2e_worker(rank, world_size, port):
    """End-to-end EP training step (real Eagle3Model TTT forward, synthetic data)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    ep_group = dist.group.WORLD

    from torchspec import AutoEagle3DraftModel
    from torchspec.models.eagle3 import Eagle3Model, compute_lazy_target_padded
    from torchspec.training.ep_utils import sync_gradients_ep
    from torchspec.training.optimizer import BF16Optimizer

    H, V, length = 128, 256, 4
    cfg = _tiny_deepseek_config()
    cfg.ep_group = ep_group
    draft = AutoEagle3DraftModel.from_config(cfg, attention_backend="sdpa", torch_dtype=DT).to(dev)
    with torch.no_grad():
        for _n, p in draft.named_parameters():
            if not getattr(p, "_is_ep", False):
                dist.broadcast(p.data, src=0)
    model = Eagle3Model(draft_model=draft, length=length, attention_backend="sdpa",
                        gradient_checkpointing=False).to(dev)
    opt = BF16Optimizer(draft, lr=1e-3, max_grad_norm=1.0, total_steps=10, warmup_ratio=0.0, ep_group=ep_group)
    assert opt.has_ep and sum(opt.ep_mask) > 0

    B, S = 2, 12
    torch.manual_seed(100 + rank)
    input_ids = torch.randint(0, V, (B, S), device=dev)
    target = compute_lazy_target_padded(
        torch.randn(B, S, H, device=dev, dtype=DT), torch.randn(V, H, device=dev, dtype=DT), length
    )
    before = draft.midlayer.mlp.experts.gate_up_proj.detach().clone()
    losses = []
    grad_norms = []
    for _step in range(2):
        opt.zero_grad()
        plosses, *_ = model(
            input_ids=input_ids, attention_mask=torch.ones(B, S, device=dev, dtype=torch.long),
            target=target, loss_mask=torch.ones(B, S, device=dev),
            hidden_states=torch.randn(B, S, H * 3, device=dev, dtype=DT), position_ids=None,
        )
        loss = sum(plosses) / len(plosses)
        loss.backward()
        sync_gradients_ep(opt.model_params, dp_group=None, ep_fsdp_group=None)
        grad_norms.append(opt.step().item())
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(x)).item() for x in losses), "non-finite loss"
    assert not torch.equal(before, draft.midlayer.mlp.experts.gate_up_proj.detach()), "experts not updated"
    # grad norm is global -> identical on all ranks
    gn = torch.tensor([grad_norms[-1]], device=dev)
    gathered = [torch.zeros_like(gn) for _ in range(world_size)]
    dist.all_gather(gathered, gn)
    assert all(abs(t.item() - grad_norms[-1]) < 1e-3 for t in gathered), "EP grad norm not consistent across ranks"
    dist.destroy_process_group()


def _ep_fsdp_worker(rank, world_size, port):
    """FSDP-on-experts (ep_size < world_size): sharded experts forward == un-sharded."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from torchspec.models.draft.moe import MoEExperts
    from torchspec.training.ep_utils import shard_experts_fsdp

    ep_size, ep_fsdp = 2, world_size // 2
    mesh = init_device_mesh("cuda", (ep_fsdp, ep_size), mesh_dim_names=("ep_fsdp", "ep"))
    ep_group = mesh.get_group("ep")
    H, isz, E, top_k, N = 128, 64, 8, 2, 40

    torch.manual_seed(7)
    ref = MoEExperts(E, H, isz, ep_group=ep_group).to(dev, DT)
    shd = MoEExperts(E, H, isz, ep_group=ep_group).to(dev, DT)
    shd.load_state_dict(ref.state_dict())
    shard_experts_fsdp(shd, mesh["ep_fsdp"])
    assert isinstance(shd.gate_up_proj.data, DTensor) or isinstance(shd.gate_up_proj, DTensor)

    torch.manual_seed(100 + rank)
    hid = torch.randn(N, H, device=dev, dtype=DT)
    rw, se = torch.randn(N, E, device=dev).sigmoid().topk(top_k, dim=-1)
    out_ref = ref(hid, rw.to(DT), se)
    out_shd = shd(hid, rw.to(DT), se)
    rel = ((out_shd.float() - out_ref.float()).abs().mean() / out_ref.float().abs().mean().clamp_min(1e-6)).item()
    assert rel < 1e-2, f"FSDP-on-experts parity rel {rel}"
    out_shd.float().pow(2).mean().backward()
    g = shd.gate_up_proj.grad
    assert g is not None and torch.isfinite(g.to_local() if isinstance(g, DTensor) else g).all()
    dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires >= 2 GPUs")
class TestMoEDraftExpertParallel(unittest.TestCase):
    def test_ep_parity_gradclip_ckpt(self):
        mp.spawn(_ep_worker, args=(2, _find_free_port()), nprocs=2, join=True)

    def test_ep_end_to_end_training_step(self):
        mp.spawn(_ep_e2e_worker, args=(2, _find_free_port()), nprocs=2, join=True)


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires >= 4 GPUs (ep=2, ep_fsdp=2)")
class TestMoEDraftFSDPExperts(unittest.TestCase):
    def test_fsdp_on_experts_parity(self):
        mp.spawn(_ep_fsdp_worker, args=(4, _find_free_port()), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
