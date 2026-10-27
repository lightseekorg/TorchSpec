"""CUDA smoke: real HFRunner -> mask resolver/collator -> Eagle3Trainer._forward.

Uses tiny random Llama target/draft networks with each checkpoint's FULL token
vocabulary and real pinned tokenizers. Observes actual embedding inputs, target
hidden states and per-depth loss masks. No tokenizer/model/forward mocks.
This qualifies plumbing, not pretrained model numerics, vLLM/SGLang or RDMA.
"""

import argparse
import json
import platform
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from tests.tokenizer_contract.support import (
    expected_mask,
    load_cases,
    load_manifest,
    resolve_checkpoint,
)
from torchspec.config.inference_config import HFInferenceConfig
from torchspec.data.renderers.smg import SmgRenderer
from torchspec.data.utils import (
    DataCollatorWithPadding,
    pack_loss_mask,
    resolve_loss_mask,
    serialize_packed_loss_mask,
)
from torchspec.inference.engine.hf_runner import HFRunner
from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from torchspec.models.eagle3 import Eagle3Model
from torchspec.models.target import HFTargetModel
from torchspec.training.eagle3_trainer import Eagle3Trainer


class ObservedEagle3(Eagle3Model):
    def _calculate_loss(self, **kwargs):
        depth = kwargs["idx"]
        expected = torch.zeros_like(self.expected_mask)
        expected[:, : expected.shape[1] - depth] = self.expected_mask[:, depth:]
        torch.testing.assert_close(kwargs["mask"], expected, rtol=0, atol=0)
        # Trainer shifts last_hidden_states once; Eagle3 slices once per depth.
        torch.testing.assert_close(
            kwargs["target"].hidden_states_padded[:, : self.expected_hs.shape[1]],
            self.expected_hs,
            rtol=0,
            atol=0,
        )
        self.observed_depths.append(depth)
        return super()._calculate_loss(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This qualification requires CUDA; absence is not a skip")
    torch.manual_seed(20261026)
    # Exercise ordinary math without paying JIT warmup for this alignment test.
    torch.compiler.set_stance("force_eager")
    results = []
    for profile, spec in load_manifest()["profiles"].items():
        path = resolve_checkpoint(profile)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        config = LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        target = LlamaForCausalLM(config).cuda().eval()
        runner = HFRunner(HFInferenceConfig(model_path=str(path)))
        runner.target_model = HFTargetModel(target)
        runner.target_model.set_aux_hidden_states_layers([0, 1, 2])
        runner._initialized = True
        draft = LlamaForCausalLMEagle3(config, attention_backend="sdpa").cuda()
        eagle = ObservedEagle3(draft, length=3, attention_backend="sdpa").cuda()
        # Invoke the real trainer forward without FSDP/optimizer/queue setup.
        trainer = object.__new__(Eagle3Trainer)
        trainer.verifier_norm = None
        trainer.t2d = None
        trainer.target_lm_head_weight = target.lm_head.weight
        trainer.eagle3 = eagle
        trainer.model = eagle
        renderer = SmgRenderer(path, spec["template"])
        cases = [c for c in load_cases(profile) if c["name"] in ("unicode", "multi_turn")]
        for last_turn in (False, True):
            consumed, norms, embedded = [], [], []
            handles = [
                target.model.embed_tokens.register_forward_pre_hook(
                    lambda module, inputs: consumed.append(inputs[0].detach().clone())
                ),
                target.model.norm.register_forward_hook(
                    lambda module, inputs, output: norms.append(output.detach().clone())
                ),
                draft.embed_tokens.register_forward_pre_hook(
                    lambda module, inputs: embedded.append(inputs[0].detach().clone())
                ),
            ]
            try:
                rendered = [
                    renderer.render(c["messages"], max_seq_length=4096, last_turn_only=last_turn)
                    for c in cases
                ]
                ids = [torch.tensor(i) for i, _ in rendered]
                masks = [
                    serialize_packed_loss_mask(pack_loss_mask(torch.tensor(m))) for _, m in rendered
                ]
                outputs = runner.generate("tokenizer-contract", ids, masks)
                assert len(consumed) == len(cases)
                features = []
                for i, (case, output) in enumerate(zip(cases, outputs, strict=True)):
                    assert consumed[i].squeeze(0).tolist() == case["input_ids"]
                    sample = output["tensors"]
                    torch.testing.assert_close(sample["input_ids"], consumed[i], rtol=0, atol=0)
                    torch.testing.assert_close(
                        sample["last_hidden_states"], norms[i], rtol=0, atol=0
                    )
                    sample["packed_loss_mask"] = output["packed_loss_mask"]
                    mask = resolve_loss_mask(sample, dynamic_loss_mask=False)
                    assert mask.tolist() == expected_mask(case, True, last_turn)
                    sample["loss_mask"] = mask.unsqueeze(0)
                    features.append(sample)
                batch = DataCollatorWithPadding()(features)
                eagle.expected_mask = batch["loss_mask"].cuda()
                eagle.expected_hs = torch.zeros_like(batch["last_hidden_states"])
                eagle.expected_hs[:, :-1] = batch["last_hidden_states"][:, 1:]
                eagle.observed_depths = []
                losses, _, _, counts, _ = trainer._forward(batch)
                assert eagle.observed_depths == [0, 1, 2]
                assert len(embedded) == 3
                for depth, observed in enumerate(embedded):
                    expected = torch.zeros_like(batch["input_ids"], device="cuda")
                    expected[:, : expected.shape[1] - depth - 1] = batch["input_ids"][
                        :, depth + 1 :
                    ].cuda()
                    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                    assert int(counts[depth]) == int(eagle.expected_mask[:, depth:].sum())
                loss = sum(losses)
                assert torch.isfinite(loss)
                loss.backward()
                assert (
                    draft.fc.weight.grad is not None and torch.isfinite(draft.fc.weight.grad).all()
                )
                assert draft.fc.weight.grad.abs().sum() > 0
                draft.zero_grad(set_to_none=True)
                results.append(
                    {
                        "profile": profile,
                        "last_turn_only": last_turn,
                        "cases": [c["name"] for c in cases],
                        "tokens": [len(c["input_ids"]) for c in cases],
                        "loss_counts": [int(n) for n in counts],
                        "status": "pass",
                    }
                )
            finally:
                for handle in handles:
                    handle.remove()
        runner.shutdown()
    Path(args.report).write_text(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "scope": "tiny random target/draft, real HFRunner and Eagle3Trainer forward/backward, direct tensors, no distributed transport",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
