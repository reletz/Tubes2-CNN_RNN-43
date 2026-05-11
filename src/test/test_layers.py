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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
