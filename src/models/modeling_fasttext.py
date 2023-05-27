from typing import Optional

import torch
from torch import Tensor
from torch.nn import init


class FasttextEmbedding(torch.nn.EmbeddingBag):

    def __init__(self, num_embeddings: int, embedding_dim: int, vocab: dict, ngrams: int,
                 special_tokens: list, pad_token_id: int,
                 **kwargs):
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         mode="sum",
                         padding_idx=pad_token_id,
                         **kwargs)
        self.word_representation = self.__compute_representations(num_embeddings, vocab,
                                                                  ngrams, special_tokens)

    @staticmethod
    def __compute_subwords(token: str, vocab, ngrams, special_tokens):
        if len(token) <= ngrams or token in special_tokens:
            return []
        return [vocab.get(token[t:t + ngrams]) for t in range(len(token) - ngrams + 1) if
                vocab.get(token[t:t + ngrams]) is not None]

    def __compute_representations(self, num_embeddings, vocab, ngrams, special_tokens):
        """

        Each word w is represented as a bag of character n-gram.

        NOTE: no special boundary symbols < and > at the beginning and end of words
        We also include the word w itself in the set of its n-grams,
        to learn a representation for each word (in addition to character n-grams).

        Taking the word where and n = 3 as an example, it will be represented by the character n-grams:
        <wh, whe, her, ere, re>, <where>

        :return:
        """

        reps = [(v, set([v] + self.__compute_subwords(k, vocab, ngrams, special_tokens))) for k, v in vocab.items()]
        max_length = len(max(reps, key=lambda x: len(x[1]))[1])
        word_representation = torch.empty((num_embeddings, max_length), dtype=torch.long)
        torch.nn.init.constant_(word_representation, self.padding_idx)
        for k, v in reps:
            word_representation[k, :len(v)] = torch.LongTensor(list(v))
        return word_representation

    def forward(self, input: Tensor, offsets: Optional[Tensor] = None,
                per_sample_weights: Optional[Tensor] = None) -> Tensor:
        # TODO fix word_representation device
        self.word_representation = self.word_representation.to(input.device)

        tokens = self.word_representation[input].reshape((-1, self.word_representation.shape[1]))
        return super().forward(tokens, offsets, per_sample_weights)


class FasttextModel(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, vocab: dict, ngrams: int,
                 special_tokens: list, pad_token_id: int, context_size: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.context_size = context_size
        self.word_embedding = FasttextEmbedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                vocab=vocab,
                                                ngrams=ngrams,
                                                special_tokens=special_tokens,
                                                pad_token_id=pad_token_id,
                                                max_norm=1,
                                                **kwargs)
        self.context_embedding = torch.nn.Embedding(num_embeddings=num_embeddings,
                                                    embedding_dim=embedding_dim,
                                                    padding_idx=pad_token_id,
                                                    max_norm=1)
        self._init_embeddings()

    def _init_embeddings(self):
        init.xavier_normal_(self.word_embedding.weight)
        init.xavier_normal_(self.context_embedding.weight)
        with torch.no_grad():
            self.context_embedding.weight[self.context_embedding.padding_idx].fill_(0)
            self.word_embedding.weight[self.context_embedding.padding_idx].fill_(0)

    def compute_context(self, input_ids: torch.LongTensor):
        return torch.cat([
            torch.cat(
                (torch.ones((input_ids.shape[0], max(self.context_size - i, 0)),
                            dtype=torch.long, device=input_ids.device) * self.context_embedding.padding_idx,
                 input_ids[:, max(i - self.context_size, 0):i].to(input_ids.device),
                 input_ids[:, i + 1:i + 1 + self.context_size].to(input_ids.device),
                 torch.ones((input_ids.shape[0], max(i - input_ids.shape[1] + self.context_size + 1,
                                                     0)), dtype=torch.long,
                            device=input_ids.device) * self.context_embedding.padding_idx),
                dim=1) for
            i in
            range(input_ids.shape[1])], dim=1).reshape((input_ids.shape[0], input_ids.shape[1], -1)).to(
            input_ids.device)

    def binary_logistic_loss(self, word, context, negative_samples):
        word = word.reshape((-1, 1, self.embedding_dim))
        positive_scores = word.bmm(context.reshape((-1, 10, self.embedding_dim)).transpose(1, 2)).reshape(
            context.shape[:3])
        negative_scores = word.bmm(
            negative_samples.reshape((word.shape[0], -1, self.embedding_dim)).transpose(1, 2)).reshape(
            negative_samples.shape[:4])
        loss = torch.log(1 + torch.e ** (-positive_scores)) + torch.log(1 + torch.e ** (negative_scores.sum(dim=-1)))
        return loss.mean()

    def forward(self, input_ids: torch.LongTensor,
                labels: Optional[torch.LongTensor] = None,
                n_samples: int = 5):
        word = self.word_embedding(input_ids).reshape((input_ids.shape[0], input_ids.shape[1], -1))
        loss = None
        if labels is not None:
            context = self.context_embedding(labels)
            negative_samples = self.context_embedding(
                torch.randint(0, self.num_embeddings, labels.shape + (n_samples,), device=input_ids.device))
            loss = self.binary_logistic_loss(word, context, negative_samples)
        return word, loss
