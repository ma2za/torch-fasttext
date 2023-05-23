from transformers import RobertaTokenizerFast, AutoModel

from src.models.modeling_fasttext import FasttextEmbedding

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
embedding = FasttextEmbedding(
    num_embeddings=tokenizer.backend_tokenizer.get_vocab_size(with_added_tokens=True),
    embedding_dim=512,
    vocab=tokenizer.backend_tokenizer.get_vocab(with_added_tokens=True))
model = AutoModel.from_pretrained("t5-small")

print()
