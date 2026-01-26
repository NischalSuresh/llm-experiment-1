import torch

from tqdm import tqdm
from accelerate import Accelerator

from src.model import get_model
from src.dataloader import get_dataloader

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def main():
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers("llm-experiment-1")
    model = get_model("HuggingFaceTB/SmolLM2-135M")
    dataloader = get_dataloader("HuggingFaceTB/SmolLM2-135M", batch_size=2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    num_training_steps = 100000
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=2000)
    cosine = CosineAnnealingLR(optimizer, T_max=num_training_steps - 2000)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[2000])
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    model.train()
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    
    log_every = 10
    
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if step % log_every == 0:
            accelerator.log({"train_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=step)
        
        if step == 100:
            break
            
    accelerator.end_training()

if __name__ == "__main__":
    main()
    print("Training completed")
