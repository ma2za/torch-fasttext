from typing import Optional

from torch import nn, Tensor
from torch.nn import functional as F


class FasttextEmbedding(nn.EmbeddingBag):

    def __init__(self, num_embeddings: int, embedding_dim: int, vocab: dict, **kwargs):
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         mode="sum", **kwargs)
        self.vocab = vocab
        self._inverted_vocab = dict(zip(vocab.values(), vocab.keys()))

    def forward(self, input: Tensor, offsets: Optional[Tensor] = None,
                per_sample_weights: Optional[Tensor] = None) -> Tensor:
        return F.embedding_bag(input, self.weight, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode, self.sparse,
                               per_sample_weights, self.include_last_offset,
                               self.padding_idx)
