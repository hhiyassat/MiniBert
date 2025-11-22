import torch
import torch.nn.functional as F
from tqdm import tqdm
import os


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, path, epoch, loss, config, global_step=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config
    }
    if global_step is not None:
        checkpoint['global_step'] = global_step
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load model, optimizer, and scheduler from checkpoint to resume training"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
    
    # Load scheduler state if available
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Scheduler state loaded")
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    global_step = checkpoint.get('global_step', None)
    
    print(f"✓ Checkpoint loaded")
    print(f"  Epoch: {epoch + 1}")
    print(f"  Loss: {loss:.4f}")
    if global_step is not None:
        print(f"  Global step: {global_step}")
    print()
    
    return model, optimizer, scheduler, epoch, loss, global_step


def train_mlm(model, train_loader, optimizer, scheduler, device, epochs, config, save_path="model/mini_bert_mlm.pt", resume_from=None, val_loader=None):
    model.train()
    best_loss = float('inf')
    best_train_loss = float('inf')  # Track best training loss separately for display
    global_step = 0
    start_epoch = 0
    check_every_batches = config['check_every_batches']
    batch_losses = []
    
    # Early stopping parameters
    patience = config.get('patience', 5)
    patience_counter = 0
    
    # Exponential Moving Average (EMA) for smoother loss tracking
    ema_decay = config.get('ema_decay', 0.99)
    ema_loss = None
    
    if resume_from is not None:
        model, optimizer, scheduler, start_epoch, saved_best_loss, saved_global_step = load_checkpoint(
            model, optimizer, scheduler, resume_from, device
        )
        
        # Restore global_step if available
        if saved_global_step is not None:
            global_step = saved_global_step
            print(f"→ Resuming from global step {global_step}")
        
        # Keep the best_loss from checkpoint to continue tracking improvement
        # This ensures we only save checkpoints that are better than previous best
        if val_loader is not None:
            # For validation-based training: keep saved validation loss to compare against
            best_loss = saved_best_loss if saved_best_loss != float('inf') else float('inf')
            # Initialize best_train_loss from saved loss as well (will be updated during training)
            best_train_loss = saved_best_loss if saved_best_loss != float('inf') else float('inf')
            print(f"→ Resuming with best validation loss: {best_loss:.4f}\n")
        else:
            # For train loss only: keep the saved best training loss
            best_loss = saved_best_loss if saved_best_loss != float('inf') else float('inf')
            best_train_loss = saved_best_loss if saved_best_loss != float('inf') else float('inf')
            print(f"→ Resuming with best loss: {best_loss:.4f}\n")
        
        start_epoch += 1
        
        if start_epoch >= epochs:
            print(f"Model already trained for {start_epoch} epochs (target: {epochs}). No additional training needed.")
            return model
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            
            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1
            global_step += 1
            batch_losses.append(loss_value)
            
            # Update EMA loss for smoother tracking
            if ema_loss is None:
                ema_loss = loss_value
            else:
                ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss_value
            
            # Update best training loss for display (always track this)
            if ema_loss < best_train_loss:
                best_train_loss = ema_loss
            
            # For display: always show best_train_loss during training (updated in real-time)
            # best_loss (validation-based) is only updated at epoch end when validation is available
            display_best = best_train_loss
            
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'ema': f'{ema_loss:.4f}',
                'best': f'{display_best:.4f}' if display_best != float('inf') else 'inf',
                'lr': f'{current_lr:.2e}',
                'step': global_step
            })
            
            # Check every N batches using smoothed EMA loss and recent average
            # Only save based on training loss if validation set is not available
            if global_step % check_every_batches == 0 and val_loader is None:
                # Calculate average of last N batches for stability
                recent_losses = batch_losses[-check_every_batches:]
                avg_recent_loss = sum(recent_losses) / len(recent_losses)
                
                # Use the minimum of EMA and recent average for more stable comparison
                comparison_loss = min(ema_loss, avg_recent_loss)
                
                # Only save if loss improved significantly (relative improvement > 0.1%)
                improvement_threshold = 0.001
                if comparison_loss < best_loss * (1 - improvement_threshold):
                    best_loss = comparison_loss
                    save_checkpoint(model, optimizer, scheduler, save_path, epoch, comparison_loss, config, global_step)
                    print(f"\n  → Step {global_step}: New best loss {comparison_loss:.4f} (EMA: {ema_loss:.4f}, Recent: {avg_recent_loss:.4f})! Model saved.")
        
        avg_loss = total_loss / num_batches
        
        # Evaluate on validation set if available
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total_loss = 0
            val_num_batches = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    val_labels = val_batch['labels'].to(device)
                    
                    val_logits = model(val_input_ids, val_attention_mask)
                    val_loss_batch = F.cross_entropy(
                        val_logits.view(-1, val_logits.size(-1)),
                        val_labels.view(-1),
                        ignore_index=-100
                    )
                    val_total_loss += val_loss_batch.item()
                    val_num_batches += 1
            
            if val_num_batches > 0:
                val_loss = val_total_loss / val_num_batches
                
                # Use validation loss for checkpointing (better for detecting overfitting)
                improvement_threshold = 0.001
                if val_loss < best_loss * (1 - improvement_threshold):
                    best_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, save_path, epoch, val_loss, config, global_step)
                    print(f"  ✓ Epoch {epoch+1}: New best validation loss {val_loss:.4f}! Model saved.")
                    patience_counter = 0  # Reset patience counter
                else:
                    # No improvement in validation loss
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n  ⚠ Early stopping triggered! No improvement in validation loss for {patience} epochs.")
                        print(f"  Best validation loss: {best_loss:.4f}")
                        print(f"  Stopping training at epoch {epoch+1}/{epochs}\n")
                        break
            
            model.train()
        
        if val_loss is not None:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_loss:.4f}, Best Train Loss: {best_train_loss:.4f}, EMA Loss: {ema_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}, Best Loss: {best_loss:.4f}, EMA Loss: {ema_loss:.4f}")
        print()
    
    return model
