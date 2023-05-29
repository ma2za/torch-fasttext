from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
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

dataset = load_dataset("emotion")

optimizer = AdamW(model.parameters(),
                  lr=0.01,
                  betas=(0.9, 0.999),
                  eps=1e-08,
                  weight_decay=0.00)
model.train()

dataloader = DataLoader(dataset["train"], batch_size=4, shuffle=False)

for batch in dataloader:
    optimizer.zero_grad()
    batch = tokenizer(batch["text"], truncation=True, padding=True, max_length=30, return_tensors="pt")
    labels = model.compute_context(batch["input_ids"])
    _, loss = model(batch["input_ids"], labels)
    print(loss)
    loss.backward()
    optimizer.step()

print()
