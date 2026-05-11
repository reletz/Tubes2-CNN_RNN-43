"""Unit tests for model builders."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module(module_name: str, file_path: str):
    """Load a module from file path to avoid package import issues."""
    spec = importlib.util.spec_from_file_location(
        module_name, str(Path(file_path).resolve())
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


cnn_model_module = _load_module("cnn_model", "src/nn/models/cnn_model.py")
caption_model_module = _load_module("caption_model", "src/nn/models/caption_model.py")


class TestCNNClassifier:
    """Smoke tests for CNNClassifier."""

    def test_forward_conv_flatten_shape(self):
        """Test a small Conv2D -> Pool -> Flatten CNN forward pass."""
        np.random.seed(42)

        model = cnn_model_module.CNNClassifier(
            input_shape=(16, 16, 3),
            num_classes=6,
            conv_blocks=(
                cnn_model_module.ConvBlockSpec(filters=4, kernel_size=(3, 3), padding="same"),
                cnn_model_module.ConvBlockSpec(filters=8, kernel_size=(3, 3), padding="same"),
            ),
            conv_kind="conv",
            pool_kind="max",
            head_kind="flatten",
        )

        model.set_conv_weights(
            0,
            np.random.randn(3, 3, 3, 4).astype(np.float32),
            np.random.randn(4).astype(np.float32),
        )
        model.set_conv_weights(
            1,
            np.random.randn(3, 3, 4, 8).astype(np.float32),
            np.random.randn(8).astype(np.float32),
        )
        model.set_output_weights(
            np.random.randn(128, 6).astype(np.float32),
            np.random.randn(6).astype(np.float32),
        )

        x = np.random.randn(2, 16, 16, 3).astype(np.float32)
        out = model.forward(x)

        assert out.shape == (2, 6)
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_forward_locallyconnected_globalavg_shape(self):
        """Test a small LocallyConnected2D -> AvgPool -> GlobalAvg CNN forward pass."""
        np.random.seed(7)

        model = cnn_model_module.CNNClassifier(
            input_shape=(8, 8, 1),
            num_classes=3,
            conv_blocks=(
                cnn_model_module.ConvBlockSpec(
                    filters=2,
                    kernel_size=(3, 3),
                    padding="valid",
                    pool_size=(2, 2),
                ),
            ),
            conv_kind="locally",
            pool_kind="avg",
            head_kind="global_avg",
        )

        out_h = (8 - 3) // 1 + 1
        out_w = (8 - 3) // 1 + 1
        model.set_conv_weights(
            0,
            np.random.randn(out_h, out_w, 3, 3, 1, 2).astype(np.float32),
            np.random.randn(out_h, out_w, 2).astype(np.float32),
        )
        model.set_output_weights(
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(3).astype(np.float32),
        )

        x = np.random.randn(1, 8, 8, 1).astype(np.float32)
        out = model.forward(x)

        assert out.shape == (1, 3)
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_invalid_input_shape_raises(self):
        """Test CNNClassifier raises for invalid input shape."""
        model = cnn_model_module.CNNClassifier(
            input_shape=(8, 8, 1),
            num_classes=2,
            conv_blocks=(cnn_model_module.ConvBlockSpec(filters=2, kernel_size=(3, 3)),),
        )
        model.set_conv_weights(
            0,
            np.random.randn(3, 3, 1, 2).astype(np.float32),
            np.random.randn(2).astype(np.float32),
        )
        model.set_output_weights(
            np.random.randn(18, 2).astype(np.float32),
            np.random.randn(2).astype(np.float32),
        )

        with pytest.raises(ValueError, match="Expected input shape"):
            model.forward(np.random.randn(8, 8).astype(np.float32))


class TestImageCaptioner:
    """Smoke tests for ImageCaptioner."""

    def test_forward_rnn_shape(self):
        """Test RNN captioner output shape."""
        np.random.seed(11)

        cap = caption_model_module.ImageCaptioner(
            vocab_size=12,
            embed_dim=7,
            hidden_units=5,
            decoder_kind="rnn",
            num_recurrent_layers=2,
        )
        cap.set_embedding_weights(np.random.randn(12, 7).astype(np.float32))
        cap.set_image_projection_weights(
            np.random.randn(10, 7).astype(np.float32), np.random.randn(7).astype(np.float32)
        )
        cap.set_output_weights(
            np.random.randn(5, 12).astype(np.float32), np.random.randn(12).astype(np.float32)
        )
        cap.set_recurrent_layer_weights(
            0,
            np.random.randn(7, 5).astype(np.float32),
            np.random.randn(5, 5).astype(np.float32),
            np.random.randn(5).astype(np.float32),
        )
        cap.set_recurrent_layer_weights(
            1,
            np.random.randn(5, 5).astype(np.float32),
            np.random.randn(5, 5).astype(np.float32),
            np.random.randn(5).astype(np.float32),
        )

        image_features = np.random.randn(3, 10).astype(np.float32)
        token_ids = np.random.randint(0, 12, size=(3, 4)).astype(np.int32)

        out = cap.forward(image_features, token_ids)

        assert out.shape == (3, 5, 12)

    def test_forward_lstm_shape(self):
        """Test LSTM captioner output shape."""
        np.random.seed(13)

        cap = caption_model_module.ImageCaptioner(
            vocab_size=8,
            embed_dim=6,
            hidden_units=4,
            decoder_kind="lstm",
            num_recurrent_layers=1,
        )
        cap.set_embedding_weights(np.random.randn(8, 6).astype(np.float32))
        cap.set_image_projection_weights(
            np.random.randn(9, 6).astype(np.float32), np.random.randn(6).astype(np.float32)
        )
        cap.set_output_weights(
            np.random.randn(4, 8).astype(np.float32), np.random.randn(8).astype(np.float32)
        )
        cap.set_recurrent_layer_weights(
            0,
            np.random.randn(6, 16).astype(np.float32),
            np.random.randn(4, 16).astype(np.float32),
            np.random.randn(16).astype(np.float32),
        )

        image_features = np.random.randn(2, 9).astype(np.float32)
        token_ids = np.random.randint(0, 8, size=(2, 3)).astype(np.int32)

        out = cap.forward(image_features, token_ids)

        assert out.shape == (2, 4, 8)

    def test_missing_output_weights_raises(self):
        """Test captioner raises if output weights are missing."""
        cap = caption_model_module.ImageCaptioner(
            vocab_size=5,
            embed_dim=4,
            hidden_units=3,
            decoder_kind="rnn",
            num_recurrent_layers=1,
        )
        cap.set_embedding_weights(np.random.randn(5, 4).astype(np.float32))
        cap.set_image_projection_weights(
            np.random.randn(7, 4).astype(np.float32), np.random.randn(4).astype(np.float32)
        )
        cap.set_recurrent_layer_weights(
            0,
            np.random.randn(4, 3).astype(np.float32),
            np.random.randn(3, 3).astype(np.float32),
            np.random.randn(3).astype(np.float32),
        )

        image_features = np.random.randn(1, 7).astype(np.float32)
        token_ids = np.random.randint(0, 5, size=(1, 2)).astype(np.int32)

        with pytest.raises(ValueError, match="Output weights are not set"):
            cap.forward(image_features, token_ids)
