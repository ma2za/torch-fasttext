from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


class FasttextEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, vocab: dict = None, ngrams: int = None,
                 special_tokens: list = None, pad_token_id: int = 1, reduce: bool = True):
        super().__init__()
        self.padding_idx = pad_token_id
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce = reduce
        if reduce:
            self.word_representation = self.__compute_representations(num_embeddings, vocab,
                                                                      ngrams, special_tokens)

        self.weight = torch.nn.Sequential(OrderedDict(
            [
                ("lr_embedding", torch.nn.Embedding(num_embeddings, 32)),
                ("A", torch.nn.Linear(32, int(embedding_dim / 2), bias=False)),
                ("B", torch.nn.Linear(int(embedding_dim / 2), embedding_dim, bias=False)),
                ("layer_norm", torch.nn.LayerNorm(embedding_dim))
            ]
        ))

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.weight.parameters():
            if param.dim() == 1:
                torch.nn.init.xavier_normal_(param.reshape(1, -1))
            else:
                torch.nn.init.xavier_normal_(param)
        with torch.no_grad():
            self.weight.lr_embedding.weight[self.padding_idx].fill_(0)

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

    def forward(self, input_ids: Tensor) -> Tensor:
        # TODO fix word_representation device
        if self.reduce:
            self.word_representation = self.word_representation.to(input_ids.device)

            tokens = self.word_representation[input_ids].reshape((-1, self.word_representation.shape[1]))
        else:
            tokens = input_ids
        embedding = self.weight(tokens)
        return embedding.sum(dim=1) if self.reduce else embedding


class FasttextModel(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, vocab: dict, ngrams: int,
                 special_tokens: list, pad_token_id: int, context_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.context_size = context_size
        self.word_embedding = FasttextEmbedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                vocab=vocab,
                                                ngrams=ngrams,
                                                special_tokens=special_tokens,
                                                pad_token_id=pad_token_id,
                                                reduce=True)
        self.context_embedding = FasttextEmbedding(num_embeddings=num_embeddings,
                                                   embedding_dim=embedding_dim,
                                                   pad_token_id=pad_token_id,
                                                   reduce=False)

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

    def score(self, query, keys):
        query = query.reshape((-1, 1, self.embedding_dim))
        scores = query.bmm(keys.reshape((query.shape[0], -1, self.embedding_dim)).transpose(1, 2))
        return scores.reshape(keys.shape[:-1])

    def binary_logistic_loss(self, word, context, negative_samples, inputs_mask, labels_mask):
        mask = inputs_mask.unsqueeze(2) * labels_mask
        positive_scores = self.score(word, context)
        negative_scores = self.score(word, negative_samples).sum(dim=-1)
        loss = (-F.logsigmoid(positive_scores) - F.logsigmoid(-negative_scores)) * mask
        return loss.sum() / mask.sum()

    def forward(self, input_ids: torch.LongTensor,
                labels: Optional[torch.LongTensor] = None,
                n_samples: int = 5,
                attention_mask=None):
        word = self.word_embedding(input_ids).reshape((input_ids.shape[0], input_ids.shape[1], -1))
        loss = None
        if labels is not None:
            context = self.context_embedding(labels)
            negative_samples = self.context_embedding(
                torch.randint(0, self.num_embeddings, labels.shape + (n_samples,), device=input_ids.device))
            loss = self.binary_logistic_loss(word, context, negative_samples,
                                             inputs_mask=(input_ids != 1).long(),
                                             labels_mask=(labels != 1).long())
        return word, loss
