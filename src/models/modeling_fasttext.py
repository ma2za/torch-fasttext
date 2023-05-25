from typing import Optional, List

import torch
from torch import Tensor


class FasttextEmbedding(torch.nn.EmbeddingBag):

    def __init__(self, num_embeddings: int, embedding_dim: int, vocab: dict, ngrams: int,
                 special_tokens: list, pad_token_id: int,
                 **kwargs):
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         mode="sum",
                         padding_idx=pad_token_id,
                         **kwargs)
        self.vocab = vocab
        self.ngrams = ngrams
        self.special_tokens = special_tokens
        self.inverted_vocab = dict(zip(vocab.values(), vocab.keys()))

    def _compute_representation(self, token: str) -> List[str]:
        """

        Each word w is represented as a bag of character n-gram.

        NOTE: no special boundary symbols < and > at the beginning and end of words
        We also include the word w itself in the set of its n-grams,
        to learn a representation for each word (in addition to character n-grams).

        Taking the word where and n = 3 as an example, it will be represented by the character n-grams:
        <wh, whe, her, ere, re>, <where>

        :param token:
        :return:
        """
        if len(token) <= self.ngrams or token in self.special_tokens:
            return [token]
        return [token[t:t + self.ngrams] for t in range(len(token) - self.ngrams + 1)] + [token]

    def _encode(self, tokens: List[List[str]], pad: bool = True) -> List[List[str]]:
        """

        :param tokens:
        :param pad:
        :return:
        """
        encoded_input = []
        for word_list in tokens:
            encoded_input.append([self.vocab[w] for w in word_list if self.vocab.get(w) is not None])
        if pad:
            max_length = len(max(encoded_input, key=lambda x: len(x)))
            for word_list in encoded_input:
                word_list.extend([self.padding_idx] * (max_length - len(word_list)))
        return encoded_input

    def forward(self, input: Tensor, offsets: Optional[Tensor] = None,
                per_sample_weights: Optional[Tensor] = None) -> Tensor:
        tokens = [self._compute_representation(self.inverted_vocab[i.cpu().item()]) for i in input.flatten()]
        tokens = torch.tensor(self._encode(tokens), dtype=input.dtype).to(input.device)
        return super().forward(tokens)


class FasttextModel(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, vocab: dict, ngrams: int,
                 special_tokens: list, pad_token_id: int, **kwargs):
        super().__init__(**kwargs)
        self.word_embedding = FasttextEmbedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                vocab=vocab,
                                                ngrams=ngrams,
                                                special_tokens=special_tokens,
                                                pad_token_id=pad_token_id,
                                                **kwargs)
        self.context_embedding = torch.nn.Embedding(num_embeddings=num_embeddings,
                                                    embedding_dim=embedding_dim,
                                                    padding_idx=pad_token_id)

    def compute_context(self, input_ids: torch.LongTensor, context_size: int):
        return torch.cat([
            torch.cat(
                (torch.ones((input_ids.shape[0], max(context_size - i, 0)),
                            dtype=torch.long) * self.context_embedding.padding_idx,
                 input_ids[:, max(i - context_size, 0):i],
                 input_ids[:, i + 1:i + 1 + context_size],
                 torch.ones((input_ids.shape[0], max(i - input_ids.shape[1] + context_size + 1,
                                                     0)), dtype=torch.long) * self.context_embedding.padding_idx),
                dim=1) for
            i in
            range(input_ids.shape[1])])

    def forward(self, input_ids: torch.LongTensor,
                labels: Optional[torch.LongTensor] = None):
        word = self.word_embedding(input_ids).reshape((input_ids.shape[0], input_ids.shape[1], -1))
        if labels is not None:
            context = self.context_embedding(labels)

        # TODO implement loss
        return word
