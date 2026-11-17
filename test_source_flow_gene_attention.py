"""Architecture and row-separability checks for the source-flow backbone."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

try:
    from .metric import calibration_rank_correlations, fixed_rbf_mmd_squared
    from .source_flow import (
        GENE_ATTENTION_ARCHITECTURE,
        LEGACY_BATCH_ATTENTION_ARCHITECTURE,
        build_source_flow_model,
        conditional_flow_matching_loss,
        load_source_flow_checkpoint,
        source_flow_architecture,
        source_flow_model_config_from_checkpoint,
    )
    from .source_flow_gene_attention import (
        GeneAttention,
        GeneAttentionSourceFlowBackbone,
    )
except ImportError:
    from metric import calibration_rank_correlations, fixed_rbf_mmd_squared
    from source_flow import (
        GENE_ATTENTION_ARCHITECTURE,
        LEGACY_BATCH_ATTENTION_ARCHITECTURE,
        build_source_flow_model,
        conditional_flow_matching_loss,
        load_source_flow_checkpoint,
        source_flow_architecture,
        source_flow_model_config_from_checkpoint,
    )
    from source_flow_gene_attention import GeneAttention, GeneAttentionSourceFlowBackbone


def small_gene_attention_config() -> dict[str, object]:
    return {
        "architecture": GENE_ATTENTION_ARCHITECTURE,
        "input_size": 5,
        "condition_size": 5,
        "hidden_size": 12,
        "depth": 2,
        "num_heads": 3,
        "mlp_ratio": 2.0,
        "time_embedding_scale": 1000.0,
    }


def randomize_parameters(model: torch.nn.Module, seed: int = 7) -> None:
    """Make every path non-trivial; DiT output/gates start at exactly zero."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.12, generator=generator)


def passing_source_performance() -> dict[str, object]:
    return {
        "passed": True,
        "validation_cfm_loss": 0.5,
        "relative_mse": 0.8,
        "median_SPCC": 0.2,
        "quality_thresholds": {
            "max_relative_mse": 1.0,
            "min_median_SPCC": 0.0,
        },
        "context": {"evaluation_domain": "source_validation"},
    }


class GeneAttentionArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_gene_attention_config()
        self.model = build_source_flow_model(self.config).eval()
        randomize_parameters(self.model)

    def test_builder_selects_gene_attention_explicitly(self) -> None:
        self.assertEqual(self.model.architecture, GENE_ATTENTION_ARCHITECTURE)
        self.assertTrue(self.model.row_separable)
        self.assertIsInstance(
            self.model.backbone, GeneAttentionSourceFlowBackbone
        )
        self.assertEqual(self.model.backbone.attention_axis, "genes")
        self.assertTrue(
            any(
                key.startswith("backbone.blks.0.attn.qkv.")
                for key in self.model.state_dict()
            )
        )

    def test_sdpa_matches_explicit_scaled_softmax_attention(self) -> None:
        attention = GeneAttention(hidden_size=12, num_heads=3).eval()
        randomize_parameters(attention, seed=11)
        tokens = torch.randn(2, 5, 12, generator=torch.Generator().manual_seed(13))

        actual = attention(tokens)
        batch, genes, hidden = tokens.shape
        qkv = attention.qkv(tokens).reshape(
            batch, genes, 3, attention.num_heads, attention.head_size
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = torch.softmax(
            (query @ key.transpose(-2, -1)) * attention.scale,
            dim=-1,
        )
        expected = attention.proj(
            (weights @ value).transpose(1, 2).reshape(batch, genes, hidden)
        )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_one_spot_does_not_depend_on_other_batch_rows(self) -> None:
        generator = torch.Generator().manual_seed(19)
        anchor_x = torch.randn(1, 5, generator=generator)
        anchor_condition = torch.randn(1, 5, generator=generator)
        peers_x = torch.randn(4, 5, generator=generator)
        peers_condition = torch.randn(4, 5, generator=generator)

        single = self.model(
            anchor_x,
            torch.tensor([0.37]),
            anchor_condition,
        )
        combined = self.model(
            torch.cat((anchor_x, peers_x), dim=0),
            torch.tensor([0.37, 0.01, 0.29, 0.68, 0.93]),
            torch.cat((anchor_condition, peers_condition), dim=0),
        )
        torch.testing.assert_close(single[0], combined[0], rtol=1e-6, atol=1e-6)

    def test_batch_permutation_and_chunking_do_not_change_outputs(self) -> None:
        generator = torch.Generator().manual_seed(23)
        x = torch.randn(7, 5, generator=generator)
        condition = torch.randn(7, 5, generator=generator)
        times = torch.rand(7, generator=generator)
        expected = self.model(x, times, condition)

        permutation = torch.tensor([5, 2, 0, 6, 1, 4, 3])
        permuted = self.model(
            x[permutation], times[permutation], condition[permutation]
        )
        torch.testing.assert_close(
            permuted, expected[permutation], rtol=1e-6, atol=1e-6
        )

        chunked = torch.cat(
            (
                self.model(x[:2], times[:2], condition[:2]),
                self.model(x[2:6], times[2:6], condition[2:6]),
                self.model(x[6:], times[6:], condition[6:]),
            ),
            dim=0,
        )
        torch.testing.assert_close(chunked, expected, rtol=1e-6, atol=1e-6)

    def test_cross_spot_input_gradients_are_zero(self) -> None:
        generator = torch.Generator().manual_seed(29)
        x = torch.randn(4, 5, generator=generator, requires_grad=True)
        condition = torch.randn(4, 5, generator=generator)
        times = torch.rand(4, generator=generator)
        self.model(x, times, condition)[1].sum().backward()

        self.assertGreater(float(x.grad[1].abs().sum()), 0.0)
        torch.testing.assert_close(x.grad[0], torch.zeros_like(x.grad[0]))
        torch.testing.assert_close(x.grad[2:], torch.zeros_like(x.grad[2:]))

    def test_flow_matching_training_path_backpropagates(self) -> None:
        train_model = build_source_flow_model(self.config)
        target = torch.randn(3, 5)
        condition = torch.randn(3, 5)
        loss, details = conditional_flow_matching_loss(
            train_model, target, condition
        )
        loss.backward()

        self.assertEqual(tuple(details["predicted_velocity"].shape), (3, 5))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in train_model.parameters()
            )
        )

    def test_gene_aligned_multichannel_condition_is_supported(self) -> None:
        config = small_gene_attention_config()
        config["condition_size"] = 10
        model = build_source_flow_model(config)
        output = model(
            torch.randn(3, 5),
            torch.rand(3),
            torch.randn(3, 10),
        )
        self.assertEqual(tuple(output.shape), (3, 5))
        self.assertEqual(model.backbone.condition_channels, 2)


class SourceFlowCheckpointArchitectureTests(unittest.TestCase):
    def test_untagged_gene_checkpoint_is_inferred_from_state_dict(self) -> None:
        config = small_gene_attention_config()
        model = build_source_flow_model(config)
        untagged = dict(config)
        del untagged["architecture"]
        resolved = source_flow_model_config_from_checkpoint(
            {
                "model_config": untagged,
                "model_state_dict": model.state_dict(),
            }
        )
        self.assertEqual(resolved["architecture"], GENE_ATTENTION_ARCHITECTURE)

    def test_untagged_gene_checkpoint_loads_strictly_and_passes_boundary_gate(
        self,
    ) -> None:
        config = small_gene_attention_config()
        original = build_source_flow_model(config)
        untagged = dict(config)
        del untagged["architecture"]
        checkpoint = {
            "method": "source_flow",
            "model_config": untagged,
            "model_state_dict": original.state_dict(),
            "source_performance": passing_source_performance(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gene.pt"
            torch.save(checkpoint, path)
            sampler, restored, loaded = load_source_flow_checkpoint(
                str(path),
                device=torch.device("cpu"),
                require_row_separable=True,
            )

        self.assertTrue(sampler.row_separable)
        self.assertTrue(restored.row_separable)
        self.assertEqual(
            loaded["model_config"]["architecture"],
            GENE_ATTENTION_ARCHITECTURE,
        )
        for expected, actual in zip(
            original.state_dict().values(), restored.state_dict().values()
        ):
            torch.testing.assert_close(actual, expected)

    def test_declared_architecture_must_match_state_dict(self) -> None:
        model = build_source_flow_model(small_gene_attention_config())
        wrong_config = small_gene_attention_config()
        wrong_config["architecture"] = LEGACY_BATCH_ATTENTION_ARCHITECTURE
        with self.assertRaisesRegex(ValueError, "metadata disagrees"):
            source_flow_model_config_from_checkpoint(
                {
                    "model_config": wrong_config,
                    "model_state_dict": model.state_dict(),
                }
            )

    def test_missing_architecture_defaults_to_legacy_batch_attention(self) -> None:
        self.assertEqual(
            source_flow_architecture({}), LEGACY_BATCH_ATTENTION_ARCHITECTURE
        )

    def test_boundary_gate_rejects_legacy_checkpoint_before_model_import(self) -> None:
        checkpoint = {
            "method": "source_flow",
            "model_config": {
                "architecture": LEGACY_BATCH_ATTENTION_ARCHITECTURE,
                "input_size": 5,
                "hidden_size": 12,
            },
            "model_state_dict": {
                "backbone.in_layer.0.weight": torch.zeros(12, 5)
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "row-separable"):
                load_source_flow_checkpoint(
                    str(path),
                    device=torch.device("cpu"),
                    require_row_separable=True,
                )


class GeneLocalPreprocessingTests(unittest.TestCase):
    def test_source_augmentation_is_opt_in(self) -> None:
        try:
            import numpy as np
            from anndata import AnnData
            try:
                from .Dataloader import preprocess_pair
            except ImportError:
                from Dataloader import preprocess_pair
        except ModuleNotFoundError as error:
            self.skipTest(f"Preprocessing dependencies unavailable: {error}")

        source = AnnData(np.array([[0, 2], [3, 4], [5, 6]], dtype=np.float32))
        spatial = AnnData(np.array([[1, 2], [3, 0]], dtype=np.float32))
        default = preprocess_pair(source, spatial, method="stdiff", noise_std=10.0)
        augmented = preprocess_pair(source, spatial, method="stdiff", augment_source=True)

        self.assertEqual(default.seq_array.shape, source.shape)
        self.assertEqual(augmented.seq_array.shape[0], 2 * source.n_obs)
        np.testing.assert_array_equal(augmented.seq_array[:source.n_obs], default.seq_array)
        shifted_log = np.log1p(source.X + 10.0)
        np.testing.assert_allclose(
            augmented.seq_array[source.n_obs:], shifted_log / (1.0 + shifted_log)
        )
        np.testing.assert_array_equal(augmented.spatial_array, default.spatial_array)
        self.assertEqual(source.X[0, 0], 0.0)

    def test_hidden_raw_genes_cannot_change_kept_condition(self) -> None:
        try:
            import numpy as np
            from anndata import AnnData
            try:
                from .Dataloader import (
                    GENE_LOCAL_PREPROCESSING,
                    LEGACY_PREPROCESSING,
                    preprocess_pair,
                )
            except ImportError:
                from Dataloader import (
                    GENE_LOCAL_PREPROCESSING,
                    LEGACY_PREPROCESSING,
                    preprocess_pair,
                )
        except ModuleNotFoundError as error:
            self.skipTest(f"Preprocessing dependencies unavailable: {error}")

        def expression(values):
            adata = AnnData(np.asarray(values, dtype=np.float32))
            adata.var_names = ["keep", "pseudo", "missing"]
            adata.obs_names = ["spot_a", "spot_b"]
            return adata

        source = expression([[1, 2, 3], [4, 5, 6]])
        original = expression([[4, 8, 12], [2, 3, 5]])
        changed = expression([[4, 800, 1200], [2, 300, 500]])
        safe_a = preprocess_pair(
            source, original, method="stdiff", preprocessing=GENE_LOCAL_PREPROCESSING
        )
        safe_b = preprocess_pair(
            source, changed, method="stdiff", preprocessing=GENE_LOCAL_PREPROCESSING
        )
        np.testing.assert_array_equal(safe_a.spatial_array[:, 0], safe_b.spatial_array[:, 0])

        old_a = preprocess_pair(
            source, original, method="stdiff", preprocessing=LEGACY_PREPROCESSING
        )
        old_b = preprocess_pair(
            source, changed, method="stdiff", preprocessing=LEGACY_PREPROCESSING
        )
        self.assertFalse(np.array_equal(old_a.spatial_array[:, 0], old_b.spatial_array[:, 0]))


class FixedPathVelocityTests(unittest.TestCase):
    def test_gauge_removes_rotation_and_retains_translation(self) -> None:
        import numpy as np

        try:
            from .flow_path_comparison import _compare_time
        except ImportError:
            from flow_path_comparison import _compare_time

        # Uniform angles give zero weak moments for the tangential rotation.
        angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        validation_angles = angles + np.pi / 128
        validation = np.column_stack((np.cos(validation_angles), np.sin(validation_angles)))
        states = np.concatenate((points, validation))
        rotation = np.column_stack((-states[:, 1], states[:, 0]))
        translation = np.broadcast_to([1.0, -2.0], states.shape)
        no_waves = np.empty((0, 2))
        result = _compare_time(
            states, rotation, states, translation, len(points), no_waves, no_waves, 1e-9
        )

        self.assertAlmostEqual(result["half_squared_velocity_gap"], 2.5, places=7)
        self.assertLess(result["source_weak_residual_rms"], 1e-8)
        self.assertLess(result["target_weak_residual_rms"], 1e-8)

    def test_calibrated_velocity_uses_the_baseline_comparison_points(self) -> None:
        import numpy as np

        try:
            from .flow_path_comparison import _compare_time
        except ImportError:
            from flow_path_comparison import _compare_time

        states = np.array([[-2.0], [-1.0], [1.0], [2.0]] * 2)
        shifted_states = states + 10.0
        no_waves = np.empty((0, 1))
        scores = _compare_time(
            states, states, states, np.zeros_like(states), 4, no_waves, no_waves, 1e-12,
            calibrated_source=(shifted_states, shifted_states),
        )

        # Both source fields are u(x)=x. Moving the midpoint would change the score.
        self.assertAlmostEqual(scores["half_squared_velocity_gap"], 1.25, places=7)
        self.assertAlmostEqual(scores["calibrated_half_squared_velocity_gap"], 1.25, places=7)
        self.assertAlmostEqual(scores["half_squared_velocity_gap_reduction"], 0.0, places=7)


class CalibrationRankMetricTests(unittest.TestCase):
    def test_endpoint_mse_averages_evaluation_draws_before_clipping(self) -> None:
        try:
            from .flow_path_comparison import _endpoint_mse
        except ImportError:
            from flow_path_comparison import _endpoint_mse

        # The fit draw is excluded, and the two evaluation draws average to -1.
        samples = np.array([[5.0], [-3.0], [1.0]])
        self.assertEqual(_endpoint_mse(samples, np.array([0.0]), fit_samples=1), 0.0)

    def test_mmd_keeps_diagonal_and_uses_the_requested_bandwidth(self) -> None:
        source = np.array([[0.0]])
        target = np.array([[2.0]])
        self.assertAlmostEqual(
            fixed_rbf_mmd_squared(source, target, bandwidth=2.0),
            2 * (1 - np.exp(-0.5)),
        )
        self.assertEqual(fixed_rbf_mmd_squared(source + 2.0, target, bandwidth=2.0), 0.0)

    def test_rank_coefficients_use_both_requested_deltas(self) -> None:
        deltas = pd.DataFrame({
            "delta_mmd_squared": [0.0, 0.3, 0.1],
            "delta_path_loss": [-1.0, -4.0, -2.0],
            "delta_mse": [-1.0, 3.0, 0.0],
        })
        result = calibration_rank_correlations(deltas)
        mmd = "spearman_delta_mmd_squared_vs_delta_mse"
        path = "spearman_delta_path_loss_vs_delta_mse"
        self.assertAlmostEqual(result[mmd], 1.0)
        self.assertAlmostEqual(result[path], -1.0)

        two_candidates = calibration_rank_correlations(deltas.iloc[:2])
        self.assertIsNone(two_candidates[mmd])
        self.assertEqual(two_candidates["undefined_reasons"][mmd], "fewer_than_three_candidates")

        constant_mmd = calibration_rank_correlations(deltas.assign(delta_mmd_squared=0.0))
        self.assertIsNone(constant_mmd[mmd])
        self.assertEqual(constant_mmd["undefined_reasons"][mmd], "constant_delta")
        self.assertAlmostEqual(constant_mmd[path], -1.0)


if __name__ == "__main__":
    unittest.main()
