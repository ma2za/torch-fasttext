from transformers import RobertaTokenizerFast

from src.models.modeling_fasttext import FasttextModel

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model = FasttextModel(
    num_embeddings=tokenizer.backend_tokenizer.get_vocab_size(with_added_tokens=True),
    embedding_dim=512,
    vocab=tokenizer.backend_tokenizer.get_vocab(with_added_tokens=True),
    ngrams=3,
    special_tokens=tokenizer.all_special_tokens,
    pad_token_id=tokenizer.pad_token_id)
inputs = tokenizer("Hello world! How are you all, my friends?", return_tensors="pt")
labels = model.compute_context(inputs["input_ids"], 5)
model(inputs["input_ids"], labels)
print()
