# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest
import torch

from tests.functional_tests.checkpoint_robustness.router_diagnostics import (
    compare_glm_router_captures,
    summarize_glm_router_shape_captures,
)


def _capture(router_logits, indices):
    return {
        "schema_version": 1,
        "framework": "test",
        "model_family": "glm4_moe_lite",
        "layers": {
            1: {
                "router_logits": torch.tensor(router_logits),
                "correction_bias": torch.zeros(3),
                "indices": torch.tensor(indices),
                "score_func": "sigmoid",
                "n_groups": 1,
            }
        },
    }


def test_router_report_separates_natural_floor_from_routed_tail(tmp_path, capsys):
    hf_path = tmp_path / "hf.pt"
    automodel_path = tmp_path / "automodel.pt"
    report_path = tmp_path / "report.json"
    torch.save(_capture([[4.0, 1.0, 0.0], [1.0, 0.99, 0.0]], [[0], [0]]), hf_path)
    torch.save(_capture([[3.9, 1.1, 0.0], [0.99, 1.0, 0.0]], [[0], [1]]), automodel_path)
    reference_logits = torch.tensor([[[2.0, -2.0], [20.0, -20.0]]])
    candidate_logits = reference_logits.clone()
    candidate_logits[:, 1, :] = -candidate_logits[:, 1, :]

    report = compare_glm_router_captures(
        hf_path,
        automodel_path,
        report_path,
        reference_logits=reference_logits,
        candidate_logits=candidate_logits,
    )

    assert report["route_flip_token_count"] == 1
    assert report["route_flip_token_fraction"] == pytest.approx(0.5)
    assert report["early_layer_summary"]["route_flip_token_count"] == 1
    assert report["early_layer_summary"]["flips_above_score_perturbation_bound_fraction"] == 0.0
    sign_test = report["early_layer_summary"]["correction_bias_direction_sign_test"]
    assert sign_test["added_bias_equal_count"] == 1
    assert sign_test["added_bias_greater_fraction_with_ties_split"] == pytest.approx(0.5)
    final_token_kl = report["final_token_kl"]
    assert final_token_kl["natural_agreement_floor_no_flipped_layers"]["token_count"] == 1
    assert final_token_kl["natural_agreement_floor_no_flipped_layers"]["mean_kl"] == pytest.approx(0.0)
    assert final_token_kl["routed_tail_one_or_more_flipped_layers"]["token_count"] == 1
    assert final_token_kl["routed_tail_one_or_more_flipped_layers"]["mean_kl"] > 1.0
    assert final_token_kl["by_flipped_layer_count"][0]["token_count"] == 1
    assert final_token_kl["by_flipped_layer_count"][1]["token_count"] == 1
    persisted = json.loads(report_path.read_text())
    assert persisted["schema_version"] == 2
    assert persisted["final_token_kl"]["by_flipped_layer_count"]["0"]["token_count"] == 1
    concise_log = capsys.readouterr().out
    assert "CHECKPOINT_ROUTER_DIAGNOSTICS" in concise_log
    assert '"layer_metrics"' not in concise_log


def test_router_shape_report_attributes_kl_to_sustained_self_flips(tmp_path):
    base_path = tmp_path / "base.pt"
    standalone_path = tmp_path / "standalone.pt"
    base_capture = _capture([[4.0, 1.0, 0.0], [1.0, 0.99, 0.0]], [[0], [0]])
    standalone_capture = _capture([[3.9, 1.1, 0.0], [0.99, 1.0, 0.0]], [[0], [1]])
    # Repeat the same routed layer under distinct indices to model 11-layer persistence.
    base_capture["layers"] = {layer: base_capture["layers"][1] for layer in range(1, 12)}
    standalone_capture["layers"] = {layer: standalone_capture["layers"][1] for layer in range(1, 12)}
    torch.save(base_capture, base_path)
    torch.save(standalone_capture, standalone_path)
    reference_logits = torch.tensor([[[2.0, -2.0], [20.0, -20.0]]])
    candidate_logits = reference_logits.clone()
    candidate_logits[:, 1, :] = -candidate_logits[:, 1, :]

    report = summarize_glm_router_shape_captures(
        base_path,
        standalone_path,
        reference_logits=reference_logits,
        candidate_logits=candidate_logits,
    )

    assert report["tokens_with_any_flip_count"] == 1
    assert report["tokens_with_sustained_flips_count"] == 1
    assert report["sustained_flip_final_token_kl"]["token_count"] == 1
    assert report["sustained_flip_final_token_kl"]["mean_kl"] > 1.0


def test_capture_hf_routers_supports_in_tree_minimax_m2(tmp_path):
    """The HF capture must wrap in-tree MiniMaxM2TopKRouter modules.

    MiniMax expresses routing as router.forward(hidden_states, bias) ->
    (router_logits, router_scores, top_k_index), unlike GLM's
    route_tokens_to_experts(router_logits) hook.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from tests.functional_tests.checkpoint_robustness.router_diagnostics import capture_glm_hf_routers

    config = AutoConfig.for_model(
        "minimax_m2",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rotary_dim=16,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(config).float().eval()
    capture_path = tmp_path / "hf.pt"
    # input_ids: Tensor of shape [batch, sequence] with 8 tokens.
    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    with capture_glm_hf_routers(model, capture_path):
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    payload = torch.load(capture_path, weights_only=True)
    assert payload["model_family"] == "minimax_m2"
    assert set(payload["layers"]) == {0, 1}
    layer = payload["layers"][0]
    assert layer["router_logits"].shape == (8, 8)
    assert layer["correction_bias"].shape == (8,)
    assert layer["indices"].shape == (8, 2)
    assert layer["score_func"] == "sigmoid"
    # The instance patch is removed once the capture context exits.
    assert "forward" not in model.model.layers[0].mlp.gate.__dict__


def test_minimax_capture_and_compare_roundtrip(tmp_path):
    """HF and AutoModel MiniMax captures must flow through the compare step.

    Regression: the AutoModel Gate reports MiniMax's n_expert_groups=0 and the
    compare step requires the normalized 1-group form (pipeline 64176861).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.minimax_m2.model import MiniMaxM2ForCausalLM
    from tests.functional_tests.checkpoint_robustness.router_diagnostics import (
        capture_glm_automodel_routers,
        capture_glm_hf_routers,
    )

    tiny = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rotary_dim=16,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    hf_config = AutoConfig.for_model("minimax_m2", **tiny)
    hf_model = AutoModelForCausalLM.from_config(hf_config).float().eval()

    am_config = AutoConfig.for_model("minimax_m2", torch_dtype="float32", **tiny)
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        dispatcher="torch",
        experts="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    am_model = MiniMaxM2ForCausalLM(am_config, backend=backend).eval()

    # input_ids: Tensor of shape [batch, sequence] with 8 tokens.
    input_ids = torch.randint(0, tiny["vocab_size"], (1, 8))
    hf_path, am_path, report_path = tmp_path / "hf.pt", tmp_path / "am.pt", tmp_path / "report.json"

    with capture_glm_hf_routers(hf_model, hf_path):
        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
    with capture_glm_automodel_routers(am_model, am_path):
        with torch.no_grad():
            am_logits = am_model(input_ids).logits

    am_payload = torch.load(am_path, weights_only=True)
    assert am_payload["model_family"] == "minimax_m2"
    assert all(layer["n_groups"] == 1 for layer in am_payload["layers"].values())

    report = compare_glm_router_captures(
        hf_path,
        am_path,
        report_path,
        reference_logits=hf_logits,
        candidate_logits=am_logits,
    )
    assert report["model_family"] == "minimax_m2"
    assert report["layer_count"] == 2
    assert 0.0 <= report["route_flip_token_fraction"] <= 1.0
    assert json.loads(report_path.read_text())["layer_count"] == 2
