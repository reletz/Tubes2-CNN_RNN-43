from __future__ import annotations

from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Literal

import numpy as np

try:
    from ..layers.embedding import Embedding
    from ..layers.recurrent import LSTMCell, SimpleRNNCell
except ImportError:  # pragma: no cover - fallback for direct file loading
    def _load_layer(module_name: str, file_name: str):
        layer_path = Path(__file__).resolve().parents[1] / "layers" / file_name
        spec = util.spec_from_file_location(module_name, str(layer_path))
        module = util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module

    _embedding_module = _load_layer("embedding_fallback", "embedding.py")
    _recurrent_module = _load_layer("recurrent_fallback", "recurrent.py")
    Embedding = _embedding_module.Embedding
    LSTMCell = _recurrent_module.LSTMCell
    SimpleRNNCell = _recurrent_module.SimpleRNNCell


DecoderKind = Literal["rnn", "lstm"]


@dataclass
class _LayerState:
    h: np.ndarray
    c: np.ndarray | None = None


class ImageCaptioner:
    """Pre-inject image captioning builder.

    The image feature vector is projected to embed_dim and prepended to the
    token embedding sequence before recurrent processing.

    This class is designed for inference/smoke testing and Keras weight loading.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_units: int,
        decoder_kind: DecoderKind = "lstm",
        num_recurrent_layers: int = 1,
    ) -> None:
        if num_recurrent_layers <= 0:
            raise ValueError("num_recurrent_layers must be >= 1")

        if decoder_kind not in {"rnn", "lstm"}:
            raise ValueError("decoder_kind must be 'rnn' or 'lstm'")

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_units = int(hidden_units)
        self.decoder_kind = decoder_kind
        self.num_recurrent_layers = int(num_recurrent_layers)

        self.embedding = Embedding(self.vocab_size, self.embed_dim)
        self.recurrent_layers = [self._make_recurrent_layer() for _ in range(self.num_recurrent_layers)]

        self.image_projection_kernel: np.ndarray | None = None
        self.image_projection_bias: np.ndarray | None = None
        self.output_kernel: np.ndarray | None = None
        self.output_bias: np.ndarray | None = None

    def _make_recurrent_layer(self):
        if self.decoder_kind == "rnn":
            return SimpleRNNCell(self.hidden_units)
        return LSTMCell(self.hidden_units)

    def set_embedding_weights(self, embedding_matrix: np.ndarray) -> None:
        self.embedding.set_weights(embedding_matrix)

    def set_image_projection_weights(self, kernel: np.ndarray, bias: np.ndarray) -> None:
        kernel = np.asarray(kernel, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        if kernel.ndim != 2:
            raise ValueError("image projection kernel must be 2D")
        if bias.ndim != 1:
            raise ValueError("image projection bias must be 1D")
        self.image_projection_kernel = kernel
        self.image_projection_bias = bias

    def set_output_weights(self, kernel: np.ndarray, bias: np.ndarray) -> None:
        kernel = np.asarray(kernel, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        if kernel.shape != (self.hidden_units, self.vocab_size):
            raise ValueError(
                f"Expected output kernel shape ({self.hidden_units}, {self.vocab_size}), got {kernel.shape}"
            )
        if bias.shape != (self.vocab_size,):
            raise ValueError(f"Expected output bias shape ({self.vocab_size},), got {bias.shape}")
        self.output_kernel = kernel
        self.output_bias = bias

    def set_recurrent_layer_weights(
        self,
        layer_index: int,
        kernel: np.ndarray,
        recurrent_kernel: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        self.recurrent_layers[layer_index].set_weights(kernel, recurrent_kernel, bias)

    def _project_image(self, image_features: np.ndarray) -> np.ndarray:
        if self.image_projection_kernel is None or self.image_projection_bias is None:
            raise ValueError("Image projection weights are not set")

        image_features = np.asarray(image_features, dtype=np.float32)
        if image_features.ndim != 2:
            raise ValueError(f"Expected image_features shape (batch, feature_dim), got {image_features.shape}")

        if image_features.shape[1] != self.image_projection_kernel.shape[0]:
            raise ValueError(
                "Image feature dimension mismatch: "
                f"expected {self.image_projection_kernel.shape[0]}, got {image_features.shape[1]}"
            )

        return image_features @ self.image_projection_kernel + self.image_projection_bias

    def _initialize_states(self, batch_size: int) -> list[_LayerState]:
        states: list[_LayerState] = []
        for layer in self.recurrent_layers:
            if isinstance(layer, SimpleRNNCell):
                states.append(_LayerState(h=layer.get_initial_state(batch_size)))
            else:
                h0, c0 = layer.get_initial_state(batch_size)
                states.append(_LayerState(h=h0, c=c0))
        return states

    def forward(self, image_features: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        """Run pre-inject caption decoding.

        Args:
            image_features: shape (batch, feature_dim)
            token_ids: shape (batch, seq_len) integer token ids

        Returns:
            Logits with shape (batch, seq_len + 1, vocab_size)
        """
        token_ids = np.asarray(token_ids, dtype=np.int32)
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token_ids shape (batch, seq_len), got {token_ids.shape}")

        batch_size = token_ids.shape[0]
        projected_image = self._project_image(image_features)  # (batch, embed_dim)
        token_embeddings = self.embedding.forward(token_ids)  # (batch, seq_len, embed_dim)

        # Pre-inject sequence: x_-1 = image projection, followed by token embeddings.
        # sequence is a list of (batch, embed_dim) arrays.
        sequence = [projected_image] + [token_embeddings[:, t, :] for t in range(token_embeddings.shape[1])]

        states = self._initialize_states(batch_size)
        outputs: list[np.ndarray] = []

        for x_t in sequence:
            current_input = x_t
            for idx, layer in enumerate(self.recurrent_layers):
                state = states[idx]
                if isinstance(layer, SimpleRNNCell):
                    state.h = layer.forward(current_input, state.h)
                    current_input = state.h
                else:
                    assert state.c is not None
                    state.h, state.c = layer.forward(current_input, state.h, state.c)
                    current_input = state.h

            if self.output_kernel is None or self.output_bias is None:
                raise ValueError("Output weights are not set")

            # current_input is the output of the last recurrent layer: (batch, hidden_units)
            logits = current_input @ self.output_kernel + self.output_bias
            outputs.append(logits)

        return np.stack(outputs, axis=1)
