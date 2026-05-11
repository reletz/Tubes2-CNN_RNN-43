from __future__ import annotations

import numpy as np


class Embedding:
    """Embedding layer: maps integer token IDs to dense vectors.

    Input shape: (batch_size, seq_len) with integer token IDs.
    Output shape: (batch_size, seq_len, embed_dim).
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """Initialize Embedding layer.

        Args:
            vocab_size: Size of vocabulary (number of unique tokens).
            embed_dim: Dimension of embedding vectors.
        """
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.embedding_matrix: np.ndarray | None = None

    def set_weights(self, embedding_matrix: np.ndarray) -> None:
        """Load embedding matrix from Keras weights.

        Args:
            embedding_matrix: Shape (vocab_size, embed_dim) with dtype float32.
        """
        embedding_matrix = np.asarray(embedding_matrix, dtype=np.float32)
        if embedding_matrix.shape != (self.vocab_size, self.embed_dim):
            raise ValueError(
                f"Expected embedding_matrix shape ({self.vocab_size}, {self.embed_dim}), "
                f"got {embedding_matrix.shape}"
            )
        self.embedding_matrix = embedding_matrix

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Lookup embeddings for token IDs.

        Args:
            x: Input token IDs, shape (batch_size, seq_len) with dtype int or int32.

        Returns:
            Embedded output, shape (batch_size, seq_len, embed_dim).
        """
        if self.embedding_matrix is None:
            raise ValueError("Embedding weights not set. Call set_weights() first.")

        x = np.asarray(x, dtype=np.int32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, seq_len), got shape {x.shape}")

        batch_size, seq_len = x.shape

        # validate token IDs
        if np.any(x < 0) or np.any(x >= self.vocab_size):
            raise ValueError(
                f"Token IDs must be in range [0, {self.vocab_size}), "
                f"got min={np.min(x)}, max={np.max(x)}"
            )

        # lookup: simple indexing
        out = self.embedding_matrix[x]  # shape (batch_size, seq_len, embed_dim)
        return out
