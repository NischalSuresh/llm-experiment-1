from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
import torch

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
        """
        Properly packs multiple sequences with:
        - EOS token separation between documents
        - Position IDs reset at document boundaries
        - Block attention masks to prevent cross-document attention
        """
        tokenized = tokenizer(examples["text"], truncation=False)
        
        concatenated_input_ids = []
        doc_boundaries = [0]
        
        for input_ids in tokenized["input_ids"]:
            concatenated_input_ids.extend(input_ids)
            concatenated_input_ids.append(tokenizer.eos_token_id)
            doc_boundaries.append(len(concatenated_input_ids))
        
        total_length = len(concatenated_input_ids)
        total_length = (total_length // seq_length) * seq_length
        
        concatenated_input_ids = concatenated_input_ids[:total_length]
        doc_boundaries = [b for b in doc_boundaries if b < total_length]
        
        num_chunks = total_length // seq_length
        input_ids_chunks = []
        position_ids_chunks = []
        attention_mask_chunks = []
        labels_chunks = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * seq_length
            end_idx = start_idx + seq_length
            
            chunk_input_ids = concatenated_input_ids[start_idx:end_idx]
            
            position_ids = []
            
            chunk_boundaries = [b - start_idx for b in doc_boundaries 
                              if start_idx <= b < end_idx]
            chunk_boundaries = [0] + chunk_boundaries + [seq_length]
            
            for i in range(len(chunk_boundaries) - 1):
                boundary_start = chunk_boundaries[i]
                boundary_end = chunk_boundaries[i + 1]
                segment_length = boundary_end - boundary_start
                position_ids.extend(range(segment_length))
            
            attention_mask = torch.ones((seq_length, seq_length), dtype=torch.bool)
            
            attention_mask = torch.tril(attention_mask)
            
            for i in range(len(chunk_boundaries) - 1):
                boundary_start = chunk_boundaries[i]
                boundary_end = chunk_boundaries[i + 1]
                
                for row in range(boundary_start, boundary_end):
                    if boundary_start > 0:
                        attention_mask[row, :boundary_start] = False
            
            # Labels are same as input_ids. Move when caculating loss
            labels = chunk_input_ids.copy()
            
            input_ids_chunks.append(chunk_input_ids)
            position_ids_chunks.append(position_ids)
            attention_mask_chunks.append(attention_mask.tolist())
            labels_chunks.append(labels)
        
        return {
            "input_ids": input_ids_chunks,
            "attention_mask": attention_mask_chunks,
            "position_ids": position_ids_chunks,
            "labels": labels_chunks
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
