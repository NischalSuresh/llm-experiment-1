from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from itertools import islice

def get_dataloader(tokenizer_name, batch_size=8, seq_length=1024):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train", 
        streaming=True
    )

    def tokenize_and_chunk(examples):
        tokenized = tokenizer(examples["text"], truncation=False)
        
        concatenated_input_ids = []
        concatenated_attention_mask = []
        
        for input_ids, attention_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            concatenated_input_ids.extend(input_ids)
            concatenated_attention_mask.extend(attention_mask)
        
        total_length = len(concatenated_input_ids)
        total_length = (total_length // seq_length) * seq_length
        
        input_ids_chunks = [
            concatenated_input_ids[i : i + seq_length] 
            for i in range(0, total_length, seq_length)
        ]
        attention_mask_chunks = [
            concatenated_attention_mask[i : i + seq_length] 
            for i in range(0, total_length, seq_length)
        ]
        
        return {
            "input_ids": input_ids_chunks,
            "attention_mask": attention_mask_chunks,
            "labels": input_ids_chunks.copy()
        }

    processed_ds = dataset.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=1000,
        remove_columns=["text", "id", "dump", "url", "date", "file_path", 
                       "language", "language_score", "token_count", "score", "int_score"]
    )

    processed_ds = processed_ds.with_format("torch")
    
    shuffled_ds = processed_ds.shuffle(buffer_size=1000, seed=42)

    dataloader = DataLoader(
        shuffled_ds,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    tokenizer_name = "gpt2"
    batch_size = 2
    seq_length = 32
    print("Testing dataloader")
    dataloader = get_dataloader(tokenizer_name, batch_size, seq_length)
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:", {k: v.shape for k, v in batch.items()})
        if i >= 2:
            break
