import unittest

import torch
import torch._dynamo as dynamo
from transformers import LlamaConfig

import torchspec.models.draft.llama3_eagle as llama_mod
from tests.utils import norm_tensor
from torchspec.models.draft.base import prepare_decoder_attention_mask
from torchspec.models.draft.llama3_eagle import (
    LlamaAttention,
    LlamaFlashAttention,
    LlamaFlexAttention,
)
from torchspec.models.ops.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
    generate_eagle3_mask,
)
from torchspec.utils.tensor import padding

dynamo.config.recompile_limit = 64
TTT_LENGTH = 7
torch.manual_seed(0)

try:
    from flash_attn import flash_attn_func as standard_flash_attn_func
except ImportError:
    standard_flash_attn_func = None


class TestFlexAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config_dict = {
            "hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "intermediate_size": 688,
            "hidden_act": "silu",
            "num_hidden_layers": 1,
            "torch_dtype": "float32",
        }
        self.config = LlamaConfig(**self.config_dict)

        self.seq_lengths = [128, 200, 256, 300, 512, 800, 1024, 2048]
        self.dtype = torch.float32

    def test_forward_pass_comparison(self):
        """Test forward pass comparison between LlamaAttention and LlamaFlexAttention."""
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_forward_pass_comparison_for_seq_len(seq_len)

    def _test_forward_pass_comparison_for_seq_len(self, seq_len):
        """Helper method to test forward pass comparison for a specific sequence length."""
        attention = LlamaAttention(self.config).to("cuda").to(self.dtype)
        flex_attention = LlamaFlexAttention(self.config).to("cuda").to(self.dtype)

        # Ensure same weights
        with torch.no_grad():
            flex_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flex_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flex_attention.v_proj.weight.copy_(attention.v_proj.weight)
            flex_attention.o_proj.weight.copy_(attention.o_proj.weight)

        attention.eval()
        flex_attention.eval()
        batch_size = 2
        hidden_size = self.config.hidden_size * 2

        ############### Attention Inputs ##############

        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        cache_keys = None
        cache_values = None
        attention_mask = torch.ones(batch_size, seq_len, dtype=self.dtype).to("cuda")
        # Simulate one item in the batch is masked and not taking a full block.
        padding_start_index = seq_len - min(200, seq_len // 3)  # Adjust padding based on seq_len
        attention_mask[1, padding_start_index:] = False
        input_embeds = norm_tensor(
            (batch_size, seq_len, self.config.hidden_size),
            device="cuda",
            dtype=self.dtype,
        )
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )
        hidden_states_list = []
        flex_hidden_states_list = []
        for idx in range(TTT_LENGTH):
            hidden_states = norm_tensor(
                (batch_size, seq_len, hidden_size), device="cuda", dtype=self.dtype
            )
            flex_hidden_states = hidden_states.clone().detach()
            hidden_states_list.append(hidden_states)
            flex_hidden_states_list.append(flex_hidden_states)

        ############### Flex Attention Inputs ##############
        flex_position_ids = position_ids.clone()
        flex_cache_keys = None
        flex_cache_values = None
        for idx in range(TTT_LENGTH):
            with torch.no_grad():
                attn_out, cache_keys, cache_values = attention(
                    hidden_states=hidden_states_list[idx],
                    attention_mask=decoder_attention_mask,
                    position_ids=position_ids,
                    cache_keys=cache_keys,
                    cache_values=cache_values,
                    use_cache=True,
                )
            with torch.no_grad():
                flex_out, flex_cache_keys, flex_cache_values = flex_attention(
                    hidden_states=flex_hidden_states_list[idx],
                    attention_mask=attention_mask,
                    position_ids=flex_position_ids,
                    cache_keys=flex_cache_keys,
                    cache_values=flex_cache_values,
                    use_cache=True,
                )
            torch.testing.assert_close(attn_out, flex_out, atol=1e-2, rtol=1e-2)

            # Check output shape
            expected_output_shape = (batch_size, seq_len, self.config.hidden_size)
            self.assertEqual(flex_out.shape, expected_output_shape)
            # Check output is not NaN or Inf
            self.assertFalse(torch.isnan(flex_out).any())
            self.assertFalse(torch.isinf(flex_out).any())

    def test_backward_pass_gradient_comparison(self):
        """Test backward pass comparing gradients between LlamaAttention and LlamaFlexAttention."""
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_backward_pass_gradient_comparison_for_seq_len(seq_len)

    def _test_backward_pass_gradient_comparison_for_seq_len(self, seq_len):
        """Helper method to test backward pass gradient comparison for a specific sequence length."""
        attention = LlamaAttention(self.config).to("cuda").to(self.dtype)
        flex_attention = LlamaFlexAttention(self.config).to("cuda").to(self.dtype)

        # Ensure same weights
        with torch.no_grad():
            flex_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flex_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flex_attention.v_proj.weight.copy_(attention.v_proj.weight)
            flex_attention.o_proj.weight.copy_(attention.o_proj.weight)

        batch_size = 2
        hidden_size = self.config.hidden_size * 2

        ############### Attention Inputs ##############
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        cache_keys = None
        cache_values = None
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to("cuda")
        # Simulate one item in the batch is masked and not taking a full block.
        # padding_start_index = seq_len - 50
        # attention_mask[1, padding_start_index:] = False
        input_embeds = norm_tensor(
            (batch_size, seq_len, self.config.hidden_size),
            device="cuda",
            dtype=self.dtype,
        )
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )

        ############### Flex Attention Inputs ##############
        flex_position_ids = position_ids.clone()
        flex_cache_keys = None
        flex_cache_values = None
        loss_mask = torch.ones(batch_size, seq_len, dtype=self.dtype, requires_grad=False).to(
            "cuda"
        )

        # Create input tensors that require gradients
        loss_list = []
        loss_flex_list = []
        hidden_states_list = []
        flex_hidden_states_list = []
        for idx in range(TTT_LENGTH):
            hidden_states = norm_tensor(
                (batch_size, seq_len, hidden_size), device="cuda", dtype=self.dtype
            )
            flex_hidden_states = hidden_states.clone().detach()
            hidden_states_list.append(hidden_states)
            flex_hidden_states_list.append(flex_hidden_states)

        for idx in range(TTT_LENGTH):
            is_last = idx == TTT_LENGTH - 1
            attn_out, cache_keys, cache_values = attention(
                hidden_states=hidden_states_list[idx],
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                cache_keys=cache_keys,
                cache_values=cache_values,
                use_cache=True,
            )
            flex_out, flex_cache_keys, flex_cache_values = flex_attention(
                hidden_states=flex_hidden_states_list[idx],
                attention_mask=attention_mask,
                position_ids=flex_position_ids,
                cache_keys=flex_cache_keys,
                cache_values=flex_cache_values,
                use_cache=True,
            )
            # Apply loss mask on calculation over batch
            loss = (attn_out * loss_mask[..., None]).sum().mean()
            loss_flex = (flex_out * loss_mask[..., None]).sum().mean()
            torch.testing.assert_close(loss, loss_flex, atol=1e-2, rtol=1e-2)
            loss_list.append(loss)
            loss_flex_list.append(loss_flex)
            # Compare gradients

            if not is_last:
                # Step 5.7: we need to update the loss mask
                loss_mask = padding(loss_mask, left=False)
        mean_loss = sum(loss_list) / len(loss_list)
        mean_loss_flex = sum(loss_flex_list) / len(loss_flex_list)
        mean_loss.backward()
        mean_loss_flex.backward()
        projections = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for proj_name in projections:
            torch.testing.assert_close(
                getattr(attention, proj_name).weight.grad,
                getattr(flex_attention, proj_name).weight.grad,
                atol=1e-2,
                rtol=1e-2,
            )


class TestEagle3FlexMask(unittest.TestCase):
    def test_eagle3_flex_mask(self):
        B = 1
        H = 1
        S = 128 * 8
        D = 128
        Q_LEN = S
        KV_LEN = S * 3
        lck = 128 * 2
        data_type = torch.bfloat16
        query = norm_tensor((B, H, S, D), device="cuda", dtype=data_type)
        key_cache = norm_tensor((B, H, KV_LEN, D), device="cuda", dtype=data_type)
        value_cache = norm_tensor((B, H, KV_LEN, D), device="cuda", dtype=data_type)
        seq_lengths = torch.tensor([S], device="cuda", dtype=torch.int32)
        seq_lengths -= lck
        block_mask = compile_friendly_create_block_mask(
            mask_mod=generate_eagle3_mask(
                seq_lengths=seq_lengths, Q_LEN=Q_LEN, KV_LEN=KV_LEN, lck=lck
            ),
            B=1,
            H=1,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=query.device,
        )
        # fmt: off
        expected_mask = torch.tensor([[[
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]]], dtype=torch.int32).to(query.device)
        # fmt: on
        dense_mask = block_mask.to_dense()
        assert torch.allclose(dense_mask, expected_mask)
        compile_friendly_flex_attention(query, key_cache, value_cache, block_mask=block_mask)


@unittest.skipUnless(standard_flash_attn_func is not None, "flash_attn not installed")
class TestFlashAttentionCachedPath(unittest.TestCase):
    def test_cached_path_gradients_match_flex_attention(self):
        cfg = LlamaConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            rms_norm_eps=1e-05,
            vocab_size=32000,
            intermediate_size=688,
            hidden_act="silu",
            num_hidden_layers=1,
            torch_dtype="bfloat16",
        )
        dtype = torch.bfloat16
        seq_len = 128
        batch_size = 2
        hidden_size = cfg.hidden_size * 2
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).repeat(batch_size, 1)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device="cuda")
        attention_mask[1, 96:] = False

        old_std_fwd = llama_mod._std_flash_attn_forward
        old_std_bwd = llama_mod._std_flash_attn_backward
        old_std_mod = llama_mod._std_flash_attn_mod
        llama_mod._std_flash_attn_mod = None

        def _standard_flash_attn_forward_wrapper(*args, **kwargs):
            kwargs.pop("window_size_left", None)
            kwargs.pop("window_size_right", None)
            kwargs.pop("return_softmax", None)
            out, lse, _ = standard_flash_attn_func(
                *args,
                return_attn_probs=True,
                **kwargs,
            )
            return out, lse, None, None

        def _standard_flash_attn_backward_wrapper(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            alibi_slopes,
            deterministic,
            rng_state,
        ):
            del dropout_p, window_size_left, window_size_right
            del softcap, alibi_slopes, deterministic, rng_state
            qh = q.permute(0, 2, 1, 3).float()
            kh = k.permute(0, 2, 1, 3).float()
            vh = v.permute(0, 2, 1, 3).float()
            if kh.shape[1] != qh.shape[1]:
                repeat = qh.shape[1] // kh.shape[1]
                kh = kh.repeat_interleave(repeat, dim=1)
                vh = vh.repeat_interleave(repeat, dim=1)
            do = dout.permute(0, 2, 1, 3).float()
            oh = out.permute(0, 2, 1, 3).float()
            lse = softmax_lse.float()

            scores = torch.matmul(qh, kh.transpose(-1, -2)) * softmax_scale
            if causal:
                q_len = scores.shape[-2]
                k_len = scores.shape[-1]
                mask = torch.triu(
                    torch.ones(q_len, k_len, device=scores.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(mask, float("-inf"))

            probs = torch.exp(scores - lse.unsqueeze(-1))
            d_v = torch.matmul(probs.transpose(-1, -2), do)
            d_p = torch.matmul(do, vh.transpose(-1, -2))
            row_dot = (do * oh).sum(dim=-1, keepdim=True)
            d_s = probs * (d_p - row_dot)
            d_q = torch.matmul(d_s, kh) * softmax_scale
            d_k = torch.matmul(d_s.transpose(-1, -2), qh) * softmax_scale
            if dk.shape[2] != d_k.shape[1]:
                kv_heads = dk.shape[2]
                repeat = d_k.shape[1] // kv_heads
                d_k = d_k.view(d_k.shape[0], kv_heads, repeat, d_k.shape[2], d_k.shape[3]).sum(
                    dim=2
                )
                d_v = d_v.view(d_v.shape[0], kv_heads, repeat, d_v.shape[2], d_v.shape[3]).sum(
                    dim=2
                )

            dq.copy_(d_q.permute(0, 2, 1, 3).to(dq.dtype))
            dk.copy_(d_k.permute(0, 2, 1, 3).to(dk.dtype))
            dv.copy_(d_v.permute(0, 2, 1, 3).to(dv.dtype))

        llama_mod._std_flash_attn_forward = _standard_flash_attn_forward_wrapper
        llama_mod._std_flash_attn_backward = _standard_flash_attn_backward_wrapper

        try:
            flex_attention = LlamaFlexAttention(cfg).to("cuda").to(dtype)
            flash_attention = LlamaFlashAttention(cfg).to("cuda").to(dtype)
            with torch.no_grad():
                for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    getattr(flash_attention, proj_name).weight.copy_(
                        getattr(flex_attention, proj_name).weight
                    )

            loss_mask = attention_mask.to(dtype)
            flex_cache_keys = flex_cache_values = None
            flash_cache_keys = flash_cache_values = None
            hidden_states_list = [
                norm_tensor((batch_size, seq_len, hidden_size), device="cuda", dtype=dtype)
                for _ in range(2)
            ]
            flex_losses = []
            flash_losses = []

            for idx in range(2):
                hidden_states = hidden_states_list[idx]
                flex_out, flex_cache_keys, flex_cache_values = flex_attention(
                    hidden_states=hidden_states.clone(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_keys=flex_cache_keys,
                    cache_values=flex_cache_values,
                    use_cache=True,
                )
                flash_out, flash_cache_keys, flash_cache_values = flash_attention(
                    hidden_states=hidden_states.clone(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_keys=flash_cache_keys,
                    cache_values=flash_cache_values,
                    use_cache=True,
                )
                flex_losses.append((flex_out * loss_mask[..., None]).sum().mean())
                flash_losses.append((flash_out * loss_mask[..., None]).sum().mean())
                if idx == 0:
                    loss_mask = torch.nn.functional.pad(loss_mask[:, 1:], (0, 1))

            (sum(flex_losses) / 2).backward()
            (sum(flash_losses) / 2).backward()

            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                torch.testing.assert_close(
                    getattr(flash_attention, proj_name).weight.grad,
                    getattr(flex_attention, proj_name).weight.grad,
                    atol=5e-2,
                    rtol=1e-2,
                    msg=f"{proj_name} grad mismatch on cached path",
                )
        finally:
            llama_mod._std_flash_attn_forward = old_std_fwd
            llama_mod._std_flash_attn_backward = old_std_bwd
            llama_mod._std_flash_attn_mod = old_std_mod


if __name__ == "__main__":
    unittest.main(verbosity=2)
