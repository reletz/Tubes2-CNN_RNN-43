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

    def copy(self) -> _LayerState:
        return _LayerState(
            h=self.h.copy(),
            c=self.c.copy() if self.c is not None else None
        )


class ImageCaptioner:
    """Pre-inject image captioning builder.

    The image feature vector is projected to embed_dim and prepended to the
    token embedding sequence before recurrent processing.
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
        if kernel.shape[0] != self.hidden_units:
            raise ValueError(f"Expected output kernel first dim {self.hidden_units}, got {kernel.shape[0]}")
        if kernel.shape[1] != self.vocab_size:
            raise ValueError(f"Expected output kernel second dim {self.vocab_size}, got {kernel.shape[1]}")
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
        """Standard forward pass (logits) for training/evaluation."""
        batch_size = token_ids.shape[0]
        projected_image = self._project_image(image_features)
        token_embeddings = self.embedding.forward(token_ids)
        sequence = [projected_image] + [token_embeddings[:, t, :] for t in range(token_embeddings.shape[1])]
        states = self._initialize_states(batch_size)
        outputs = []
        for x_t in sequence:
            curr = x_t
            for idx, layer in enumerate(self.recurrent_layers):
                state = states[idx]
                if isinstance(layer, SimpleRNNCell):
                    state.h = layer.forward(curr, state.h)
                else:
                    state.h, state.c = layer.forward(curr, state.h, state.c)
                curr = state.h
            outputs.append(curr @ self.output_kernel + self.output_bias)
        return np.stack(outputs, axis=1)

    def generate_captions(
        self,
        image_features: np.ndarray,
        start_token: int,
        end_token: int,
        method: Literal["greedy", "beam"] = "greedy",
        beam_width: int = 3,
        max_len: int = 20,
    ) -> list[list[int]]:
        """Public batch API for caption generation."""
        if method == "greedy":
            return self.greedy_decode(image_features, start_token, end_token, max_len)
        return self.beam_search_decode(image_features, start_token, end_token, beam_width, max_len)

    def greedy_decode(
        self,
        image_features: np.ndarray,
        start_token: int,
        end_token: int,
        max_len: int = 20,
    ) -> list[list[int]]:
        """Vectorized greedy decoding for batches."""
        batch_size = image_features.shape[0]
        curr = self._project_image(image_features)
        states = self._initialize_states(batch_size)
        
        # Step -1: Image
        for idx, layer in enumerate(self.recurrent_layers):
            state = states[idx]
            if isinstance(layer, SimpleRNNCell): state.h = layer.forward(curr, state.h)
            else: state.h, state.c = layer.forward(curr, state.h, state.c)
            curr = state.h
            
        last_tokens = np.full((batch_size,), start_token, dtype=np.int32)
        results = [[] for _ in range(batch_size)]
        finished = np.zeros(batch_size, dtype=bool)

        for _ in range(max_len):
            if np.all(finished): break
            curr = self.embedding.forward(last_tokens[:, None])[:, 0, :]
            for idx, layer in enumerate(self.recurrent_layers):
                state = states[idx]
                if isinstance(layer, SimpleRNNCell): state.h = layer.forward(curr, state.h)
                else: state.h, state.c = layer.forward(curr, state.h, state.c)
                curr = state.h
            
            logits = curr @ self.output_kernel + self.output_bias
            next_tokens = np.argmax(logits, axis=1)
            for i in range(batch_size):
                if not finished[i]:
                    token = int(next_tokens[i])
                    if token == end_token: finished[i] = True
                    else: results[i].append(token)
            last_tokens = next_tokens
        return results

    def beam_search_decode(
        self,
        image_features: np.ndarray,
        start_token: int,
        end_token: int,
        beam_width: int = 3,
        max_len: int = 20,
    ) -> list[list[int]]:
        """Batch beam search (internally loops over single images)."""
        return [self._beam_search_single(image_features[i:i+1], start_token, end_token, beam_width, max_len) 
                for i in range(image_features.shape[0])]

    def _beam_search_single(self, image_features, start_token, end_token, beam_width, max_len):
        curr = self._project_image(image_features)
        states = self._initialize_states(1)
        for idx, layer in enumerate(self.recurrent_layers):
            state = states[idx]
            if isinstance(layer, SimpleRNNCell): state.h = layer.forward(curr, state.h)
            else: state.h, state.c = layer.forward(curr, state.h, state.c)
            curr = state.h

        beams = [(0.0, [], [s.copy() for s in states])]
        completed = []
        for _ in range(max_len):
            new_beams = []
            for score, seq, b_states in beams:
                last_token = seq[-1] if seq else start_token
                curr = self.embedding.forward(np.array([[last_token]]))[:, 0, :]
                c_states = [s.copy() for s in b_states]
                for idx, layer in enumerate(self.recurrent_layers):
                    state = c_states[idx]
                    if isinstance(layer, SimpleRNNCell): state.h = layer.forward(curr, state.h)
                    else: state.h, state.c = layer.forward(curr, state.h, state.c)
                    curr = state.h
                
                logits = curr @ self.output_kernel + self.output_bias
                log_probs = logits[0] - np.max(logits) - np.log(np.sum(np.exp(logits - np.max(logits))))
                top_idx = np.argsort(log_probs)[-beam_width:]
                for idx in top_idx:
                    n_score, n_seq = score + log_probs[idx], seq + [int(idx)]
                    if idx == end_token: completed.append((n_score, n_seq))
                    else: new_beams.append((n_score, n_seq, [s.copy() for s in c_states]))
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]
            if not beams: break
        
        all_res = completed + [(s, q) for s, q, _ in beams]
        if not all_res: return []
        all_res.sort(key=lambda x: x[0], reverse=True)
        return all_res[0][1]


class InitInjectCaptioner(ImageCaptioner):
    """Alternative init-inject architecture."""
    def __init__(self, *args, merge_mode: Literal["add", "concat"] = "add", **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_mode = merge_mode

    def set_output_weights(self, kernel, bias):
        expected = self.hidden_units * (2 if self.merge_mode == "concat" else 1)
        if kernel.shape[0] != expected: raise ValueError(f"Expected {expected}, got {kernel.shape[0]}")
        self.output_kernel, self.output_bias = kernel, bias

    def forward(self, image_features, token_ids):
        batch_size = token_ids.shape[0]
        proj_img = self._project_image(image_features)
        token_embs = self.embedding.forward(token_ids)
        states = self._initialize_states(batch_size)
        outputs = []
        for t in range(token_embs.shape[1]):
            curr = token_embs[:, t, :]
            for idx, layer in enumerate(self.recurrent_layers):
                state = states[idx]
                if isinstance(layer, SimpleRNNCell): state.h = layer.forward(curr, state.h)
                else: state.h, state.c = layer.forward(curr, state.h, state.c)
                curr = state.h
            merged = (curr + proj_img) if self.merge_mode == "add" else np.concatenate([curr, proj_img], axis=1)
            outputs.append(merged @ self.output_kernel + self.output_bias)
        return np.stack(outputs, axis=1)

    def greedy_decode(self, image_features, start_token, end_token, max_len=20):
        batch_size = image_features.shape[0]
        proj_img = self._project_image(image_features)
        states = self._initialize_states(batch_size)
        last_tokens = np.full((batch_size,), start_token, dtype=np.int32)
        results, finished = [[] for _ in range(batch_size)], np.zeros(batch_size, dtype=bool)
        for _ in range(max_len):
            if np.all(finished): break
            curr = self.embedding.forward(last_tokens[:, None])[:, 0, :]
            for idx, layer in enumerate(self.recurrent_layers):
                state = states[idx]
                if isinstance(layer, SimpleRNNCell): state.h = layer.forward(curr, state.h)
                else: state.h, state.c = layer.forward(curr, state.h, state.c)
                curr = state.h
            merged = (curr + proj_img) if self.merge_mode == "add" else np.concatenate([curr, proj_img], axis=1)
            logits = merged @ self.output_kernel + self.output_bias
            next_tokens = np.argmax(logits, axis=1)
            for i in range(batch_size):
                if not finished[i]:
                    token = int(next_tokens[i])
                    if token == end_token: finished[i] = True
                    else: results[i].append(token)
            last_tokens = next_tokens
        return results

    def _beam_search_single(self, image_features, start_token, end_token, beam_width, max_len):
        proj_img = self._project_image(image_features)
        states = self._initialize_states(1)
        beams, completed = [(0.0, [], [s.copy() for s in states])], []
        for _ in range(max_len):
            new_beams = []
            for score, seq, b_states in beams:
                last_token = seq[-1] if seq else start_token
                curr = self.embedding.forward(np.array([[last_token]]))[:, 0, :]
                c_states = [s.copy() for s in b_states]
                for idx, layer in enumerate(self.recurrent_layers):
                    state = c_states[idx]
                    if isinstance(layer, SimpleRNNCell): state.h = layer.forward(curr, state.h)
                    else: state.h, state.c = layer.forward(curr, state.h, state.c)
                    curr = state.h
                merged = (curr + proj_img) if self.merge_mode == "add" else np.concatenate([curr, proj_img], axis=1)
                logits = merged @ self.output_kernel + self.output_bias
                log_probs = logits[0] - np.max(logits) - np.log(np.sum(np.exp(logits - np.max(logits))))
                top_idx = np.argsort(log_probs)[-beam_width:]
                for idx in top_idx:
                    n_score, n_seq = score + log_probs[idx], seq + [int(idx)]
                    if idx == end_token: completed.append((n_score, n_seq))
                    else: new_beams.append((n_score, n_seq, [s.copy() for s in c_states]))
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]
            if not beams: break
        all_res = completed + [(s, q) for s, q, _ in beams]
        if not all_res: return []
        all_res.sort(key=lambda x: x[0], reverse=True)
        return all_res[0][1]
