from torch.optim import AdamW
from transformers import RobertaTokenizerFast

from src.models.modeling_fasttext import FasttextModel

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model = FasttextModel(
    num_embeddings=tokenizer.backend_tokenizer.get_vocab_size(with_added_tokens=True),
    embedding_dim=512,
    vocab=tokenizer.backend_tokenizer.get_vocab(with_added_tokens=True),
    ngrams=3,
    special_tokens=tokenizer.all_special_tokens,
    pad_token_id=tokenizer.pad_token_id,
    context_size=5)

inputs = tokenizer.batch_encode_plus([
    "Hello world! How are you all, my friends?",
    "This is a brand new world my dear friend"
], padding=True, return_tensors="pt")
labels = model.compute_context(inputs["input_ids"])
optimizer = AdamW(model.parameters(),
                  lr=0.01,
                  betas=(0.9, 0.999),
                  eps=1e-08,
                  weight_decay=0.00)
model.train()
for i in range(100000):
    model.zero_grad()
    _, loss = model(inputs["input_ids"], labels)
    print(loss)
    loss.backward()
    optimizer.step()

print()
