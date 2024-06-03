import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel
from config import *
from preprocessing import preprocess, collate_fn

def train_step(model, batch, optimizer):
    optimizer.zero_grad()
    outputs = model(pixel_values=batch['pixel_values'], labels=batch['labels'])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model(model, valid_loader, rank):
    model.eval()
    valid_loss = 0.0
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in valid_loader:
            pixel_values = batch["pixel_values"].to(rank)
            label_ids = batch["labels"].to(rank)
            outputs = model(pixel_values=pixel_values, labels=label_ids)
            loss = outputs.loss
            valid_loss += loss.item()

            logits = outputs.logits.detach().cpu()
            predictions.extend(logits.argmax(dim=-1).tolist())
            labels.extend(label_ids.tolist())

    avg_val_loss = valid_loss / len(valid_loader)
    return avg_val_loss, predictions, labels

    
def train_model(device, world_size, train_dataloader, valid_loader, batch_size=32, num_epochs=10, learning_rate=1e-3, patience=3, weight_decay=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    best_val_loss = float('inf')
    stop_counter = 0  # Counter for early stopping

    train_losses = []

    val_losses = []
    prev_lr = optimizer.param_groups[0]['lr']  # Get initial learning rate

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        train_dataloader_iter = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for i, data in enumerate(train_dataloader_iter):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            loss = train_step(model, batch, optimizer, pixel_values, labels)
            train_loss += loss

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss, predictions, labels = evaluate_model(model, valid_loader, device)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
        eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_prediction)
        
        # bleu_values.append(metrics['bleu'])
        # rouge1_values.append(metrics['rouge'])
        # rouge2_values.append(metrics['rouge2'])
        # rougeL_values.append(metrics['rougeL'])
        # spice.append(metrics['spice'])
        
        print(f"\n BLEU: {metrics['bleu']:.4f}, " +
            f"ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-2: {metrics['rouge2']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}," +
                f"SPICE: {metrics['spice']:.4f}\n")

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate

        # Print the learning rate only if it changes
        if current_lr != prev_lr:
            print("Learning rate changed to:", current_lr)
            prev_lr = current_lr  # Update previous learning rate

        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pt')
            checkpoint_path = f"./image-captioning/checkpoint-{epoch + 1}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            image_processor.save_pretrained(checkpoint_path)

            best_val_loss = avg_val_loss
            stop_counter = 0
        else:
            stop_counter += 1

        # Early stopping
        if stop_counter >= patience:
            print("Early stopping...")
            break

    print("\n---Finished Training---\n")
