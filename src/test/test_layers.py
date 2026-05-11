"""Unit tests for neural network layers."""

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
    spec.loader.exec_module(mod)
    return mod


# Load modules
conv_module = _load_module("conv", "src/nn/layers/conv.py")
pool_module = _load_module("pool", "src/nn/layers/pooling.py")
flat_module = _load_module("flat", "src/nn/layers/flatten.py")
emb_module = _load_module("emb", "src/nn/layers/embedding.py")
recurrent_module = _load_module("recurrent", "src/nn/layers/recurrent.py")


class TestConv2D:
    """Test Conv2D layer."""

    def test_forward_valid_padding(self):
        """Test Conv2D forward with valid padding."""
        conv = conv_module.Conv2D(filters=2, kernel_size=(3, 3), padding="valid")
        kernel = np.ones((3, 3, 1, 2), dtype=np.float32)
        bias = np.zeros(2, dtype=np.float32)
        conv.set_weights(kernel, bias)

        x = np.ones((1, 5, 5, 1), dtype=np.float32)
        out = conv.forward(x)

        assert out.shape == (1, 3, 3, 2), f"Expected (1,3,3,2), got {out.shape}"
        assert np.allclose(out, 9.0), "Conv output should be 9.0 (3*3*1)"

    def test_forward_same_padding(self):
        """Test Conv2D forward with same padding."""
        conv = conv_module.Conv2D(filters=1, kernel_size=(3, 3), padding="same")
        kernel = np.ones((3, 3, 3, 1), dtype=np.float32)
        bias = np.zeros(1, dtype=np.float32)
        conv.set_weights(kernel, bias)

        x = np.ones((1, 4, 4, 3), dtype=np.float32)
        out = conv.forward(x)

        assert out.shape == (1, 4, 4, 1), f"Expected (1,4,4,1), got {out.shape}"

    def test_forward_batch(self):
        """Test Conv2D with batch > 1."""
        conv = conv_module.Conv2D(filters=4, kernel_size=(3, 3), padding="valid")
        kernel = np.random.randn(3, 3, 3, 4).astype(np.float32)
        bias = np.random.randn(4).astype(np.float32)
        conv.set_weights(kernel, bias)

        x = np.random.randn(5, 8, 8, 3).astype(np.float32)
        out = conv.forward(x)

        assert out.shape == (5, 6, 6, 4), f"Expected (5,6,6,4), got {out.shape}"

    def test_forward_strides(self):
        """Test Conv2D with strides > 1."""
        conv = conv_module.Conv2D(
            filters=3, kernel_size=(3, 3), strides=(2, 2), padding="valid"
        )
        kernel = np.random.randn(3, 3, 1, 3).astype(np.float32)
        bias = np.zeros(3, dtype=np.float32)
        conv.set_weights(kernel, bias)

        x = np.random.randn(1, 8, 8, 1).astype(np.float32)
        out = conv.forward(x)

        assert out.shape == (1, 3, 3, 3), f"Expected (1,3,3,3), got {out.shape}"

    def test_set_weights_required(self):
        """Test that forward fails without set_weights."""
        conv = conv_module.Conv2D(filters=1, kernel_size=(3, 3))
        x = np.ones((1, 5, 5, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="weights are not set"):
            conv.forward(x)


class TestLocallyConnected2D:
    """Test LocallyConnected2D layer."""

    def test_forward_valid_padding(self):
        """Test LocallyConnected2D forward with valid padding."""
        lc = conv_module.LocallyConnected2D(
            kernel_size=(3, 3), filters=2, padding="valid"
        )

        # output shape with valid padding for 5x5 input, 3x3 kernel, stride 1
        out_h = (5 - 3) // 1 + 1  # 3
        out_w = (5 - 3) // 1 + 1  # 3

        kernel = np.ones((out_h, out_w, 3, 3, 1, 2), dtype=np.float32)
        bias = np.zeros((out_h, out_w, 2), dtype=np.float32)
        lc.set_weights(kernel, bias)

        x = np.ones((1, 5, 5, 1), dtype=np.float32)
        out = lc.forward(x)

        assert out.shape == (1, 3, 3, 2), f"Expected (1,3,3,2), got {out.shape}"

    def test_forward_flattened_kernel(self):
        """Test LocallyConnected2D with flattened kernel shape."""
        lc = conv_module.LocallyConnected2D(
            kernel_size=(3, 3), filters=2, padding="valid"
        )

        out_h = 3
        out_w = 3
        kernel = np.ones((out_h * out_w, 3 * 3 * 1, 2), dtype=np.float32)
        bias = np.zeros((out_h * out_w, 2), dtype=np.float32)
        lc.set_weights(kernel, bias)

        x = np.ones((1, 5, 5, 1), dtype=np.float32)
        out = lc.forward(x)

        assert out.shape == (1, 3, 3, 2), f"Expected (1,3,3,2), got {out.shape}"

    def test_forward_batch(self):
        """Test LocallyConnected2D with batch > 1."""
        lc = conv_module.LocallyConnected2D(
            kernel_size=(3, 3), filters=3, padding="valid"
        )

        out_h = 3
        out_w = 3
        kernel = np.random.randn(out_h, out_w, 3, 3, 3, 3).astype(np.float32)
        bias = np.zeros((out_h, out_w, 3), dtype=np.float32)
        lc.set_weights(kernel, bias)

        x = np.random.randn(4, 5, 5, 3).astype(np.float32)
        out = lc.forward(x)

        assert out.shape == (4, 3, 3, 3), f"Expected (4,3,3,3), got {out.shape}"


class TestPooling:
    """Test pooling layers."""

    def test_maxpool2d_basic(self):
        """Test MaxPool2D basic functionality."""
        mp = pool_module.MaxPool2D(pool_size=(2, 2))

        x = np.arange(1 * 4 * 4 * 1).reshape(1, 4, 4, 1).astype(np.float32)
        out = mp.forward(x)

        assert out.shape == (1, 2, 2, 1), f"Expected (1,2,2,1), got {out.shape}"

    def test_maxpool2d_preserves_channels(self):
        """Test MaxPool2D preserves number of channels."""
        mp = pool_module.MaxPool2D(pool_size=(2, 2))

        x = np.random.randn(2, 8, 8, 16).astype(np.float32)
        out = mp.forward(x)

        assert out.shape == (2, 4, 4, 16), f"Expected (2,4,4,16), got {out.shape}"

    def test_avgpool2d_basic(self):
        """Test AvgPool2D basic functionality."""
        ap = pool_module.AvgPool2D(pool_size=(2, 2))

        x = np.arange(1 * 4 * 4 * 1).reshape(1, 4, 4, 1).astype(np.float32)
        out = ap.forward(x)

        assert out.shape == (1, 2, 2, 1), f"Expected (1,2,2,1), got {out.shape}"

    def test_avgpool2d_preserves_channels(self):
        """Test AvgPool2D preserves number of channels."""
        ap = pool_module.AvgPool2D(pool_size=(2, 2))

        x = np.random.randn(3, 6, 6, 8).astype(np.float32)
        out = ap.forward(x)

        assert out.shape == (3, 3, 3, 8), f"Expected (3,3,3,8), got {out.shape}"

    def test_global_avgpool2d(self):
        """Test GlobalAvgPooling2D."""
        gap = pool_module.GlobalAvgPooling2D()

        x = np.random.randn(2, 4, 4, 8).astype(np.float32)
        out = gap.forward(x)

        assert out.shape == (2, 8), f"Expected (2,8), got {out.shape}"

    def test_global_maxpool2d(self):
        """Test GlobalMaxPooling2D."""
        gmp = pool_module.GlobalMaxPooling2D()

        x = np.random.randn(3, 5, 5, 16).astype(np.float32)
        out = gmp.forward(x)

        assert out.shape == (3, 16), f"Expected (3,16), got {out.shape}"

    def test_maxpool2d_custom_strides(self):
        """Test MaxPool2D with custom strides."""
        mp = pool_module.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        x = np.random.randn(1, 8, 8, 1).astype(np.float32)
        out = mp.forward(x)

        assert out.shape == (1, 4, 4, 1), f"Expected (1,4,4,1), got {out.shape}"


class TestFlatten:
    """Test Flatten layer."""

    def test_forward_4d(self):
        """Test Flatten forward with 4D input (N,H,W,C)."""
        fl = flat_module.Flatten()

        x = np.random.randn(2, 4, 4, 3).astype(np.float32)
        out = fl.forward(x)

        assert out.shape == (2, 4 * 4 * 3), f"Expected (2,48), got {out.shape}"

    def test_forward_3d(self):
        """Test Flatten forward with 3D input (H,W,C)."""
        fl = flat_module.Flatten()

        x = np.random.randn(4, 4, 3).astype(np.float32)
        out = fl.forward(x)

        assert out.shape == (1, 4 * 4 * 3), f"Expected (1,48), got {out.shape}"

    def test_backward_4d(self):
        """Test Flatten backward with 4D input."""
        fl = flat_module.Flatten()

        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        out = fl.forward(x)

        grad_out = np.random.randn(2, 3 * 4 * 5).astype(np.float32)
        grad_in = fl.backward(grad_out)

        assert grad_in.shape == x.shape, f"Expected {x.shape}, got {grad_in.shape}"

    def test_backward_3d(self):
        """Test Flatten backward with 3D input."""
        fl = flat_module.Flatten()

        x = np.random.randn(4, 4, 3).astype(np.float32)
        out = fl.forward(x)

        grad_out = np.random.randn(1, 4 * 4 * 3).astype(np.float32)
        grad_in = fl.backward(grad_out)

        assert grad_in.shape == x.shape, f"Expected {x.shape}, got {grad_in.shape}"

    def test_backward_without_forward_raises(self):
        """Test that backward raises error without forward."""
        fl = flat_module.Flatten()

        with pytest.raises(ValueError, match="Cannot call backward"):
            fl.backward(np.ones((2, 48)))


class TestEmbedding:
    """Test Embedding layer."""

    def test_forward_shape(self):
        """Test Embedding forward output shape."""
        vocab_size = 50
        embed_dim = 16
        layer = emb_module.Embedding(vocab_size, embed_dim)

        weights = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        layer.set_weights(weights)

        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        out = layer.forward(x)

        assert out.shape == (2, 3, 16), f"Expected (2,3,16), got {out.shape}"

    def test_forward_values(self):
        """Test Embedding lookup returns correct rows."""
        vocab_size = 10
        embed_dim = 4
        layer = emb_module.Embedding(vocab_size, embed_dim)

        weights = np.arange(vocab_size * embed_dim, dtype=np.float32).reshape(
            vocab_size, embed_dim
        )
        layer.set_weights(weights)

        x = np.array([[0, 3, 9]], dtype=np.int32)
        out = layer.forward(x)

        assert np.allclose(out[0, 0], weights[0])
        assert np.allclose(out[0, 1], weights[3])
        assert np.allclose(out[0, 2], weights[9])

    def test_invalid_token_raises(self):
        """Test Embedding raises for out-of-range token IDs."""
        vocab_size = 10
        embed_dim = 4
        layer = emb_module.Embedding(vocab_size, embed_dim)

        weights = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        layer.set_weights(weights)

        x = np.array([[0, 10]], dtype=np.int32)
        with pytest.raises(ValueError, match="Token IDs must be in range"):
            layer.forward(x)


class TestRecurrent:
    """Test recurrent layers."""

    def test_simplernn_forward_shape(self):
        """Test SimpleRNNCell forward output shape."""
        units = 5
        input_dim = 3
        batch_size = 4

        layer = recurrent_module.SimpleRNNCell(units)
        kernel = np.random.randn(input_dim, units).astype(np.float32)
        recurrent_kernel = np.random.randn(units, units).astype(np.float32)
        bias = np.random.randn(units).astype(np.float32)
        layer.set_weights(kernel, recurrent_kernel, bias)

        x = np.random.randn(batch_size, input_dim).astype(np.float32)
        h_prev = layer.get_initial_state(batch_size)
        out = layer.forward(x, h_prev)

        assert out.shape == (batch_size, units), f"Expected ({batch_size},{units}), got {out.shape}"

    def test_simplernn_initial_state(self):
        """Test SimpleRNNCell initial state."""
        layer = recurrent_module.SimpleRNNCell(7)
        state = layer.get_initial_state(3)

        assert state.shape == (3, 7)
        assert np.allclose(state, 0.0)

    def test_lstm_forward_shape(self):
        """Test LSTMCell forward output shapes."""
        units = 6
        input_dim = 4
        batch_size = 2

        layer = recurrent_module.LSTMCell(units)
        kernel = np.random.randn(input_dim, 4 * units).astype(np.float32)
        recurrent_kernel = np.random.randn(units, 4 * units).astype(np.float32)
        bias = np.random.randn(4 * units).astype(np.float32)
        layer.set_weights(kernel, recurrent_kernel, bias)

        x = np.random.randn(batch_size, input_dim).astype(np.float32)
        h_prev, c_prev = layer.get_initial_state(batch_size)
        h_out, c_out = layer.forward(x, h_prev, c_prev)

        assert h_out.shape == (batch_size, units), f"Expected ({batch_size},{units}), got {h_out.shape}"
        assert c_out.shape == (batch_size, units), f"Expected ({batch_size},{units}), got {c_out.shape}"

    def test_lstm_initial_state(self):
        """Test LSTMCell initial state."""
        layer = recurrent_module.LSTMCell(8)
        h0, c0 = layer.get_initial_state(5)

        assert h0.shape == (5, 8)
        assert c0.shape == (5, 8)
        assert np.allclose(h0, 0.0)
        assert np.allclose(c0, 0.0)

    def test_lstm_multi_step(self):
        """Test LSTMCell can process multiple timesteps."""
        units = 3
        input_dim = 4
        batch_size = 2

        layer = recurrent_module.LSTMCell(units)
        kernel = np.random.randn(input_dim, 4 * units).astype(np.float32)
        recurrent_kernel = np.random.randn(units, 4 * units).astype(np.float32)
        bias = np.random.randn(4 * units).astype(np.float32)
        layer.set_weights(kernel, recurrent_kernel, bias)

        x = np.random.randn(batch_size, input_dim).astype(np.float32)
        h_prev, c_prev = layer.get_initial_state(batch_size)

        h1, c1 = layer.forward(x, h_prev, c_prev)
        h2, c2 = layer.forward(x, h1, c1)

        assert h1.shape == (batch_size, units)
        assert c1.shape == (batch_size, units)
        assert h2.shape == (batch_size, units)
        assert c2.shape == (batch_size, units)


class TestDataflow:
    """Test end-to-end dataflow."""

    def test_conv_pool_flatten_dataflow(self):
        """Test Conv2D → MaxPool2D → Flatten dataflow."""
        np.random.seed(42)

        # Create layers
        conv = conv_module.Conv2D(filters=4, kernel_size=(3, 3), padding="valid")
        kernel = np.random.randn(3, 3, 3, 4).astype(np.float32)
        bias = np.zeros(4, dtype=np.float32)
        conv.set_weights(kernel, bias)

        mp = pool_module.MaxPool2D(pool_size=(2, 2))
        fl = flat_module.Flatten()

        # Input
        x = np.random.randn(2, 16, 16, 3).astype(np.float32)

        # Forward
        conv_out = conv.forward(x)
        assert conv_out.shape == (2, 14, 14, 4)

        pool_out = mp.forward(conv_out)
        assert pool_out.shape == (2, 7, 7, 4)

        flat_out = fl.forward(pool_out)
        assert flat_out.shape == (2, 7 * 7 * 4)

    def test_locally_connected_pool_flatten_dataflow(self):
        """Test LocallyConnected2D → MaxPool2D → Flatten dataflow."""
        np.random.seed(43)

        # LocallyConnected2D
        lc = conv_module.LocallyConnected2D(kernel_size=(3, 3), filters=3, padding="valid")
        out_h = (8 - 3) // 1 + 1  # 6
        out_w = (8 - 3) // 1 + 1  # 6
        kernel = np.random.randn(out_h, out_w, 3, 3, 1, 3).astype(np.float32)
        bias = np.zeros((out_h, out_w, 3), dtype=np.float32)
        lc.set_weights(kernel, bias)

        mp = pool_module.MaxPool2D(pool_size=(2, 2))
        fl = flat_module.Flatten()

        # Input
        x = np.random.randn(1, 8, 8, 1).astype(np.float32)

        # Forward
        lc_out = lc.forward(x)
        assert lc_out.shape == (1, 6, 6, 3)

        pool_out = mp.forward(lc_out)
        assert pool_out.shape == (1, 3, 3, 3)

        flat_out = fl.forward(pool_out)
        assert flat_out.shape == (1, 3 * 3 * 3)


class TestImageCaptioner:
    """Unit tests for ImageCaptioner builder."""

    def test_forward_rnn_shape(self):
        vocab_size = 12
        embed_dim = 7
        hidden_units = 5
        batch = 3
        seq_len = 4

        model = recurrent_module  # alias
        from src.nn.models.caption_model import ImageCaptioner

        cap = ImageCaptioner(vocab_size=vocab_size, embed_dim=embed_dim, hidden_units=hidden_units, decoder_kind="rnn", num_recurrent_layers=2)
        cap.set_embedding_weights(np.random.randn(vocab_size, embed_dim).astype(np.float32))
        cap.set_image_projection_weights(np.random.randn(10, embed_dim).astype(np.float32), np.random.randn(embed_dim).astype(np.float32))
        cap.set_output_weights(np.random.randn(hidden_units, vocab_size).astype(np.float32), np.random.randn(vocab_size).astype(np.float32))

        # set recurrent layer weights
        cap.set_recurrent_layer_weights(0, np.random.randn(embed_dim, hidden_units).astype(np.float32), np.random.randn(hidden_units, hidden_units).astype(np.float32), np.random.randn(hidden_units).astype(np.float32))
        cap.set_recurrent_layer_weights(1, np.random.randn(hidden_units, hidden_units).astype(np.float32), np.random.randn(hidden_units, hidden_units).astype(np.float32), np.random.randn(hidden_units).astype(np.float32))

        image_features = np.random.randn(batch, 10).astype(np.float32)
        token_ids = np.random.randint(0, vocab_size, size=(batch, seq_len)).astype(np.int32)

        out = cap.forward(image_features, token_ids)
        assert out.shape == (batch, seq_len + 1, vocab_size)

    def test_forward_lstm_shape(self):
        vocab_size = 8
        embed_dim = 6
        hidden_units = 4
        batch = 2
        seq_len = 3

        from src.nn.models.caption_model import ImageCaptioner

        cap = ImageCaptioner(vocab_size=vocab_size, embed_dim=embed_dim, hidden_units=hidden_units, decoder_kind="lstm", num_recurrent_layers=1)
        cap.set_embedding_weights(np.random.randn(vocab_size, embed_dim).astype(np.float32))
        cap.set_image_projection_weights(np.random.randn(9, embed_dim).astype(np.float32), np.random.randn(embed_dim).astype(np.float32))
        cap.set_output_weights(np.random.randn(hidden_units, vocab_size).astype(np.float32), np.random.randn(vocab_size).astype(np.float32))
        cap.set_recurrent_layer_weights(0, np.random.randn(embed_dim, 4 * hidden_units).astype(np.float32), np.random.randn(hidden_units, 4 * hidden_units).astype(np.float32), np.random.randn(4 * hidden_units).astype(np.float32))

        image_features = np.random.randn(batch, 9).astype(np.float32)
        token_ids = np.random.randint(0, vocab_size, size=(batch, seq_len)).astype(np.int32)

        out = cap.forward(image_features, token_ids)
        assert out.shape == (batch, seq_len + 1, vocab_size)

    def test_missing_output_weights_raises(self):
        from src.nn.models.caption_model import ImageCaptioner

        cap = ImageCaptioner(vocab_size=5, embed_dim=4, hidden_units=3, decoder_kind="rnn", num_recurrent_layers=1)
        cap.set_embedding_weights(np.random.randn(5, 4).astype(np.float32))
        cap.set_image_projection_weights(np.random.randn(7, 4).astype(np.float32), np.random.randn(4).astype(np.float32))
        cap.set_recurrent_layer_weights(0, np.random.randn(4, 3).astype(np.float32), np.random.randn(3, 3).astype(np.float32), np.random.randn(3).astype(np.float32))

        image_features = np.random.randn(1, 7).astype(np.float32)
        token_ids = np.random.randint(0, 5, size=(1, 2)).astype(np.int32)

        with pytest.raises(ValueError, match="Output weights are not set"):
            cap.forward(image_features, token_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
