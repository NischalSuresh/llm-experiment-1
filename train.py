import torch
from pathlib import Path

from tqdm import tqdm
from accelerate import Accelerator

from src.model import get_model
from src.dataloader import get_dataloader

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

OUTPUT_DIR = Path("checkpoints")
SAVE_MILESTONES = [1000, 5000, 10000, 25000, 50000, 100000]


def get_last_checkpoint(output_dir):
    """Find the latest checkpoint to resume from."""
    if not output_dir.exists():
        return None, 0
    
    checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
    if not checkpoints:
        return None, 0
    
    latest = max(checkpoints, key=lambda x: int(x.name.split("_")[1]))
    step = int(latest.name.split("_")[1])
    return latest / "training_state", step


def save_checkpoint(accelerator, model, step, output_dir):
    """Save checkpoint for both inference and resuming training."""
    checkpoint_dir = output_dir / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator.save_state(checkpoint_dir / "training_state")
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        checkpoint_dir / "model",
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    accelerator.print(f"Saved checkpoint to {checkpoint_dir}")


def main():
    accelerator = Accelerator(log_with="wandb", mixed_precision="bf16")
    accelerator.init_trackers("llm-experiment-1")
    model = get_model("HuggingFaceTB/SmolLM2-135M")
    dataloader = get_dataloader("HuggingFaceTB/SmolLM2-135M", batch_size=4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    max_grad_norm = 1.0  # Gradient clipping
    num_training_steps = 100000
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=2000)
    cosine = CosineAnnealingLR(optimizer, T_max=num_training_steps - 2000)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[2000])
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    resume_path, resume_step = get_last_checkpoint(OUTPUT_DIR)
    if resume_path and resume_path.exists():
        accelerator.print(f"Resuming from {resume_path} (step {resume_step})")
        accelerator.load_state(resume_path)
        starting_step = resume_step + 1
    else:
        starting_step = 0
    
    model.train()
    progress_bar = tqdm(range(num_training_steps), initial=starting_step, disable=not accelerator.is_local_main_process)
    
    log_every = 10
    
    for step, batch in enumerate(train_dataloader, start=starting_step):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if step % log_every == 0:
            accelerator.log({"train_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=step)
        
        if step in SAVE_MILESTONES:
            save_checkpoint(accelerator, model, step, OUTPUT_DIR)
    
    save_checkpoint(accelerator, model, step, OUTPUT_DIR)
    accelerator.end_training()

if __name__ == "__main__":
    main()
    print("Training completed")
