import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os
import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_path = './model_checkpoints'
epochs = 3
def calculate_accuracy(outputs, labels):
    # Assuming outputs are logits and labels are the ground-truth token IDs
    predictions = torch.argmax(outputs.logits, dim=-1)
    correct = (predictions == labels) & (labels != -100)  # Ignore padding tokens or other special tokens
    accuracy = correct.sum().float() / correct.numel()
    return accuracy.item()


def train_model(model, loader_train, loader_val, optimizer, type_model, model_name, scheduler):
    # Initialize the gradient scaler for mixed precision
    scaler = GradScaler()
    os.makedirs(model_save_path, exist_ok=True)  # Create the save directory if it doesn't exist
    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        train_progress_bar = tqdm(loader_train, desc=f"Epoch {epoch+1} Training", leave=False)

        for batch in train_progress_bar:
            optimizer.zero_grad()  # Clear previous gradients

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Mixed precision training
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()  # Scale loss to adjust for mixed precision
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping

            scaler.step(optimizer)  # Optimizer step
            scaler.update()  # Update the scaler
            scheduler.step()  # Update the learning rate

            total_train_loss += loss.item()
            acc = calculate_accuracy(outputs, labels)
            total_train_acc += acc
            train_progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch)),
                                            'training_acc': '{:.3f}'.format(acc)})
            wandb.log({"epoch": epoch, "training_loss": float(loss.item() / len(batch)), "training_acc": acc})

        avg_train_loss = total_train_loss / len(loader_train)
        avg_train_acc = total_train_acc / len(loader_train)
        print(f"\nEpoch {epoch+1} finished. Average Training Loss: {avg_train_loss:.3f}, Average Training Accuracy: {avg_train_acc:.3f}")

        # Save model checkpoint
        if epoch == 2:
            checkpoint_path = os.path.join(model_save_path, f'model_entities_epoch_{epoch+1}_{type_model}_{model_name}.pt')
            torch.save(model.state_dict(), checkpoint_path)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        val_progress_bar = tqdm(loader_val, desc=f"Epoch {epoch+1} Validation", leave=False)

        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                total_val_loss += loss.item()
                wandb.log({"epoch": epoch, "val_loss": float(loss.item() / len(batch))})

        avg_val_loss = total_val_loss / len(loader_val)
        print(f"Validation Loss: {avg_val_loss:.3f}")