from datasets import load_dataset

# requirements:
# pip install datasets pandas pyarrow

# Load IMDB (automatically handles Parquet)
ds = load_dataset("imdb", split="train")

# Save as plain text
with open("imdb.txt", "w", encoding="utf-8") as f:
    for example in ds:
        f.write(example["text"].replace("\n", " ") + "\n")

# Save as JSONL
import json
with open("imdb.jsonl", "w", encoding="utf-8") as f:
    for i, example in enumerate(ds):
        f.write(json.dumps({"id": i, "text": example["text"]}) + "\n")
